from PIL import Image, ImageDraw

def create_satellite_position_image(df, img_size=(800, 800)):
    """
    Generate a simple satellite position map as an image (no plots).

    Parameters:
    - df: DataFrame with x, y columns (positions in km)
    - img_size: size of the output image (width, height)

    Returns:
    - PIL.Image object
    """
    img = Image.new("RGB", img_size, color="black")
    draw = ImageDraw.Draw(img)

    if df.empty or not all(col in df.columns for col in ['x', 'y']):
        return img  # Empty image if no data

    # Determine scaling based on min/max x/y
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()
    margin = 10  # pixels

    def scale(val, min_val, max_val, img_min, img_max):
        if max_val == min_val:
            return (img_min + img_max) // 2
        return int((val - min_val) / (max_val - min_val) * (img_max - img_min) + img_min)

    for _, row in df.iterrows():
        x = scale(row['x'], min_x, max_x, margin, img_size[0] - margin)
        y = scale(row['y'], min_y, max_y, margin, img_size[1] - margin)
        # y is inverted in image coordinates
        y = img_size[1] - y
        draw.ellipse((x-3, y-3, x+3, y+3), fill="white")

    return img
