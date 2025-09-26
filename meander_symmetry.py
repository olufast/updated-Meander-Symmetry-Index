"""
Project Workflow

meander_symmetry.py

Compute meander symmetry index (σ) for river bends.

Modes:
- Visual mode: uses meanderlimits.shp + apex_points.shp (auto-detected from "point data/<year>/")
- Auto: curvature-based (Finotello et al. 2024, Supplementary Fig. 11). Uses inflexion (k=0 crossings) and apex = max |k| between inflexions


Features:
- Loops over adjacent year pairs and first–last pair (e.g., 2010->2013, 2013->2015, 2010->2015).

- Outputs:
    .csv         (results)
    .gpkg        (eroded polygons)
    .png         (plots)
    reflection_points.gpkg (if valley_line provided)

- Summary across all pairs:
    - meander_symmetry_summary.csv
"""

# IMPORT LIBRARIES
import os, math, argparse, warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, GeometryCollection
from shapely.ops import split

warnings.filterwarnings("ignore", category=UserWarning)

# Creating functions
def ensure_single_linestring(geom):
    if geom is None: return None
    if geom.geom_type == "LineString": return geom
    if geom.geom_type == "MultiLineString":
        parts = list(geom); parts.sort(key=lambda g: g.length, reverse=True); return parts[0]
    if geom.geom_type == "Polygon": return LineString(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        parts = list(geom); parts.sort(key=lambda g: g.area, reverse=True)
        return LineString(parts[0].exterior.coords)
    raise ValueError(f"Unsupported geometry type {geom.geom_type}")

def estimate_channel_width(ls0, ls1):
    N=200; distances=[]
    for frac in np.linspace(0,1,N):
        p=ls0.interpolate(frac*ls0.length); q=ls1.interpolate(ls1.project(p))
        distances.append(p.distance(q))
    med=np.median(distances)
    return 20.0 if med<=0.5 else max(5.0, med)

def resample_line_coords(ls, step):
    n=max(3,int(math.ceil(ls.length/step))+1)
    positions=np.linspace(0,ls.length,n)
    pts=[ls.interpolate(s) for s in positions]
    coords=np.array([[p.x,p.y] for p in pts])
    return coords,positions

def smooth_array(x,window_pts=5):
    if window_pts<=1: return x
    w=np.ones(window_pts)/window_pts; pad=window_pts//2
    xp=np.pad(x,pad_width=pad,mode='edge'); return np.convolve(xp,w,mode='valid')

def compute_curvature(ls,sample_step=None,smoothing_window=7):
    if sample_step is None: sample_step=max(ls.length/1000.0,1.0)
    coords,positions=resample_line_coords(ls,sample_step)
    x=coords[:,0]; y=coords[:,1]
    dx=np.gradient(x,positions); dy=np.gradient(y,positions)
    ddx=np.gradient(dx,positions); ddy=np.gradient(dy,positions)
    denom=(dx*dx+dy*dy)**1.5; denom[denom==0]=np.nan
    kappa=(dx*ddy-dy*ddx)/denom
    return positions,smooth_array(kappa,smoothing_window)

def find_inflections_from_curvature(positions,curvature):
    signs=np.sign(curvature); zero_cross_idx=[]
    for i in range(len(signs)-1):
        if np.isnan(signs[i]) or np.isnan(signs[i+1]): continue
        if signs[i]==0: zero_cross_idx.append(i)
        elif signs[i]*signs[i+1]<0: zero_cross_idx.append(i+1)
    idxs=[0]+sorted(set(zero_cross_idx))+[len(positions)-1]
    return [positions[i] for i in sorted(set(idxs))]

def find_apexes_between_inflections(positions,curvature,inflection_positions):
    apex_positions=[]
    for a,b in zip(inflection_positions[:-1],inflection_positions[1:]):
        mask=(positions>=a)&(positions<=b)
        if not np.any(mask): continue
        sub_idx=np.where(mask)[0]; sub_k=np.abs(curvature[sub_idx])
        if sub_k.size==0: continue
        imax=sub_idx[np.nanargmax(sub_k)]; apex_positions.append(positions[imax])
    return sorted(set(apex_positions))

def make_long_cutline_at_point(ls,pt,length=2000):
    proj=ls.project(pt); delta=min(1.0,ls.length*1e-3)
    p0=ls.interpolate(max(0.0,proj-delta)); p1=ls.interpolate(min(ls.length,proj+delta))
    dx,dy=(p1.x-p0.x,p1.y-p0.y); mag=math.hypot(dx,dy) or 1.0; ux,uy=dx/mag,dy/mag
    return LineString([(pt.x-ux*length,pt.y-uy*length),(pt.x+ux*length,pt.y+uy*length)])

def safe_split_geometry(poly,cutline):
    try: parts=split(poly,cutline)
    except: return [poly]
    if isinstance(parts,GeometryCollection): parts=list(parts.geoms)
    else:
        try: parts=list(parts)
        except: parts=[parts]
    out=[]
    for g in parts:
        if g.geom_type=="Polygon": out.append(g)
        elif g.geom_type=="MultiPolygon": out.extend(list(g.geoms))
    return out

def classify_sigma(sigma):
    if sigma is None or (isinstance(sigma,float) and np.isnan(sigma)): return "undefined"
    if sigma<0.90: return "upstream_rotation"
    if sigma<=1.05: return "extension"
    return "downstream_rotation"

# IMPLEMENTING PER-PAIR COMPUTATION
def process_pair(center_t0_path,center_t1_path,outdir,mode="visual",
                 meanderlimits_path=None,valley_line_path=None,plot=False):
    g0=gpd.read_file(center_t0_path); g1=gpd.read_file(center_t1_path)
    if g0.crs is None: raise ValueError("centerline_t0 has no CRS")
    if g1.crs!=g0.crs: g1=g1.to_crs(g0.crs)
    ls0=ensure_single_linestring(g0.geometry.unary_union)
    ls1=ensure_single_linestring(g1.geometry.unary_union)

    chan_w=estimate_channel_width(ls0,ls1)
    buff0=ls0.buffer(chan_w/2.0,cap_style=2); buff1=ls1.buffer(chan_w/2.0,cap_style=2)
    eroded_all=buff0.symmetric_difference(buff1)

    # segmentation
    if mode=="visual" and meanderlimits_path and os.path.exists(meanderlimits_path):
        ml=gpd.read_file(meanderlimits_path);
        if ml.crs!=g0.crs: ml=ml.to_crs(g0.crs)
        ml_line=ensure_single_linestring(ml.geometry.unary_union)
        proj_vals=[ls0.project(Point(x,y)) for x,y in ml_line.coords]
        proj_vals=sorted(set(proj_vals))
        segments=[(a,b,LineString([ls0.interpolate(a),ls0.interpolate(b)])) for a,b in zip(proj_vals[:-1],proj_vals[1:]) if (b-a)>1.0]
        apex_on_line=[]
        year_tag=Path(center_t0_path).stem[:4]
        apex_path=Path(center_t0_path).parents[1]/"point data"/year_tag/f"{year_tag}B1.shp"
        if apex_path.exists():
            ap=gpd.read_file(apex_path);
            if ap.crs!=g0.crs: ap=ap.to_crs(g0.crs)
            apex_on_line=[(ls0.project(p),ls0.interpolate(ls0.project(p))) for p in ap.geometry]
    else:
        positions,curvature=compute_curvature(ls0)
        inflections=find_inflections_from_curvature(positions,curvature)
        apex_positions=find_apexes_between_inflections(positions,curvature,inflections)
        segments=[(a,b,LineString([ls0.interpolate(a),ls0.interpolate(b)])) for a,b in zip(inflections[:-1],inflections[1:]) if (b-a)>1.0]
        apex_on_line=[(pos,ls0.interpolate(pos)) for pos in apex_positions]

    # reflection points (optional)
    reflection_points=[]
    if valley_line_path and os.path.exists(valley_line_path):
        vl=gpd.read_file(valley_line_path)
        if vl.crs!=g0.crs: vl=vl.to_crs(g0.crs)
        valley_geom=vl.geometry.unary_union
        for line in [ls0, ls1]:
            inter=valley_geom.intersection(line)
            if inter.is_empty: continue
            if inter.geom_type=="Point": reflection_points.append(inter)
            elif inter.geom_type=="MultiPoint": reflection_points.extend(list(inter.geoms))

    # compute areas
    results=[]; eroded_records=[]
    for idx,(a_proj,b_proj,segline) in enumerate(segments):
        strip=segline.buffer(chan_w*8.0,cap_style=2)
        eroded=eroded_all.intersection(strip)
        if eroded.is_empty:
            results.append({"meander_id":idx,"sigma":np.nan,"classification":"undefined"}); continue
        mid=0.5*(a_proj+b_proj)
        if apex_on_line: apex_proj,apex_pt=min(apex_on_line,key=lambda ap:abs(ap[0]-mid))
        else: apex_proj,apex_pt=(mid,ls0.interpolate(mid))
        cutline=make_long_cutline_at_point(ls0,apex_pt,length=max(2000.0,ls0.length*0.5))
        parts=safe_split_geometry(eroded,cutline)
        up_area=down_area=0.0
        for p in parts:
            c=p.centroid; proj=ls0.project(c)
            if proj<apex_proj: up_area+=p.area
            else: down_area+=p.area
        sigma=float("inf") if up_area==0 and down_area>0 else (down_area/up_area if up_area>0 else np.nan)
        classification=classify_sigma(sigma)
        results.append({"meander_id":idx,"up_area":up_area,"down_area":down_area,"sigma":sigma,"classification":classification})
        eroded_records.append({"meander_id":idx,"geometry":eroded,"sigma":sigma,"classification":classification})

    # save
    pair_tag=f"{Path(center_t0_path).stem}__{Path(center_t1_path).stem}"; mode_tag=mode
    out_prefix=Path(outdir)/f"{pair_tag}_{mode_tag}"; os.makedirs(outdir,exist_ok=True)
    df=pd.DataFrame(results); df.to_csv(str(out_prefix)+".csv",index=False)
    if eroded_records:
        gpd.GeoDataFrame(eroded_records,crs=g0.crs).to_file(str(out_prefix)+".gpkg",layer="eroded_meanders",driver="GPKG")

    # reflection point saving
    if reflection_points:
        ref_gdf=gpd.GeoDataFrame({"geometry":reflection_points},crs=g0.crs)
        ref_gdf.to_file(str(out_prefix)+"_reflection_points.gpkg",layer="reflection",driver="GPKG")

    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        g0.plot(ax=ax, color="blue", label="t0")
        g1.plot(ax=ax, color="red", label="t1")

        if eroded_records:
            gpd.GeoDataFrame(eroded_records, crs=g0.crs).plot(ax=ax, color="orange", alpha=0.4)

        if apex_on_line:
            ax.scatter([pt.x for _, pt in apex_on_line],[pt.y for _, pt in apex_on_line],
                       color="green", marker="x", label="Apex")

        if mode=="auto":
            positions,curvature=compute_curvature(ls0)
            inflections=find_inflections_from_curvature(positions,curvature)
            inf_pts=[ls0.interpolate(pos) for pos in inflections]
            ax.scatter([p.x for p in inf_pts],[p.y for p in inf_pts],
                       color="red", marker="o", s=30, label="Inflection")

        plt.title(f"Meander Symmetry {pair_tag} ({mode})")
        plt.legend()
        plot_path=str(out_prefix)+".png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  plot saved: {plot_path}")

    return df

# BATCH RUNNER
def run_batch(centerlines_dir,meanderlimits=None,valley_line=None,outdir="output",mode="both",plot=False):
    files=sorted([str(Path(centerlines_dir)/f) for f in os.listdir(centerlines_dir) if f.lower().endswith(".shp")])
    if len(files)<2: raise ValueError("Need >=2 centerline shapefiles")
    pairs=[(files[i],files[i+1]) for i in range(len(files)-1)]+[(files[0],files[-1])]
    all_results=[]
    for a,b in pairs:
        if mode in("visual","both"):
            df_vis=process_pair(a,b,outdir,mode="visual",meanderlimits_path=meanderlimits,valley_line_path=valley_line,plot=plot)
            df_vis["pair"]=f"{Path(a).stem}__{Path(b).stem}"; df_vis["mode"]="visual"; all_results.append(df_vis)
        if mode in("auto","both"):
            df_auto=process_pair(a,b,outdir,mode="auto",meanderlimits_path=None,valley_line_path=valley_line,plot=plot)
            df_auto["pair"]=f"{Path(a).stem}__{Path(b).stem}"; df_auto["mode"]="auto"; all_results.append(df_auto)
    if all_results:
        summary=pd.concat(all_results,ignore_index=True); os.makedirs(outdir,exist_ok=True)
        summary_path=Path(outdir)/"meander_symmetry_summary.csv"; summary.to_csv(summary_path,index=False)
        print(f"Summary CSV written: {summary_path}")
    return all_results

# COMMAND LINE INTERFACE
def main():
        p = argparse.ArgumentParser()
        p.add_argument("--centerlines_dir", required=True)
        p.add_argument("--meanderlimits", required=False)
        p.add_argument("--valley_line", required=False)
        p.add_argument("--outdir", default="output")
        p.add_argument("--mode", choices=["visual", "auto", "both"], default="both")
        p.add_argument("--plot", action="store_true")
        args = p.parse_args()
        run_batch(args.centerlines_dir, meanderlimits=args.meanderlimits,
                  valley_line=args.valley_line, outdir=args.outdir, mode=args.mode, plot=args.plot)

if __name__ == "__main__": main()

