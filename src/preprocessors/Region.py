class Region:
    def __init__(self, margin_top=0.0, margin_bottom=0.0, margin_left=0.0, margin_right=0.0):
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.margin_left = margin_left
        self.margin_right = margin_right

    def apply(self, image):
        height, width = image.shape[:2]
        top = int(height * self.margin_top)
        bottom = int(height * (1 - self.margin_bottom))
        left = int(width * self.margin_left)
        right = int(width * (1 - self.margin_right))
        print(f"Applying region: top={top}, bottom={bottom}, left={left}, right={right}, image shape={image.shape}")
        return image[top:bottom, left:right]
