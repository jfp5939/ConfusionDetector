"""Run once to generate static/icons/icon-192.png and icon-512.png."""
import os

os.makedirs("static/icons", exist_ok=True)

try:
    from PIL import Image, ImageDraw, ImageFont

    for size in [192, 512]:
        img = Image.new("RGB", (size, size), color="#0f0f0f")
        draw = ImageDraw.Draw(img)
        margin = size // 8
        draw.rounded_rectangle(
            [margin, margin, size - margin, size - margin],
            radius=size // 6,
            fill="#6c63ff",
        )
        font_size = size // 4
        font = None
        for font_path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        text = "PB"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((size - tw) // 2 - bbox[0], (size - th) // 2 - bbox[1]),
            text,
            fill="#ffffff",
            font=font,
        )
        img.save(f"static/icons/icon-{size}.png")
        print(f"Created static/icons/icon-{size}.png")

except ImportError:
    # Pillow not available — create minimal valid PNGs using raw bytes
    import struct
    import zlib

    def make_png(size, bg_rgb=(108, 99, 255)):
        """Create a solid-color PNG of the given size."""
        width = height = size
        raw_rows = []
        for _ in range(height):
            row = b"\x00" + bytes(bg_rgb) * width
            raw_rows.append(row)
        raw = b"".join(raw_rows)
        compressed = zlib.compress(raw)

        def chunk(name, data):
            c = name + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr_data)
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )
        return png

    for size in [192, 512]:
        with open(f"static/icons/icon-{size}.png", "wb") as f:
            f.write(make_png(size))
        print(f"Created static/icons/icon-{size}.png (solid purple, Pillow not installed)")

print("Done.")
