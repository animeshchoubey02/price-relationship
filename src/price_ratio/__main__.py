import argparse
from pathlib import Path
from .pipeline import run

def main():
 p=argparse.ArgumentParser(description="Learn relative price relationships from a product catalog")
 p.add_argument("catalog",nargs="?",default="data/product_catalog.csv")
 p.add_argument("--max-pairs",type=int,default=20000)
 args=p.parse_args(); r=run(args.catalog,max_pairs=args.max_pairs)
 print(f"Products: {len(r['catalog'])}")
 print(f"Pair examples: {r['pair_count']}")
 print(f"Held-out pair MAE (log ratio): {r['pairwise_test_mae']:.4f}")
 print(f"Held-out pair R²: {r['pairwise_test_r2']:.4f}")
 print(f"Graph edge RMSE: {r['graph_edge_rmse']:.4f}")
 print(f"Centered log-price RMSE (descriptive): {r['centered_log_price_rmse']:.4f}")
 print("\nLargest model feature importances:")
 for name,value in sorted(zip(r['features'],r['model'].feature_importances_),key=lambda x:x[1],reverse=True)[:10]: print(f"  {name}: {value:.4f}")
if __name__=="__main__": main()
