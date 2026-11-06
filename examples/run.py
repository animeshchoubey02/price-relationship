from price_ratio.pipeline import run

r = run("data/product_catalog.csv")
print({k: v for k, v in r.items() if isinstance(v, (int, float))})
