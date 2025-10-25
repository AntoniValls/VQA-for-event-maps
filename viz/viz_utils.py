# --- headless-safe preview helper ---
def save_preview(img, out_path="data/preview.png"):
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved preview to {out_path}")
