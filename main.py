import csv
import re
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import pytesseract


# Change this if Tesseract is installed somewhere else.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract():
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return True

    if Path(TESSERACT_PATH).is_file():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        return True

    print("Tesseract OCR was not found.")
    print("Install it or update TESSERACT_PATH near the top of this file.")
    print(f"Current path: {TESSERACT_PATH}")
    return False


def choose_images():
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select student ID card images",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return paths


def crop_card(image, filename):
    """Select the card area to reduce OCR noise from the background."""
    preview = image.copy()
    height, width = preview.shape[:2]
    scale = min(1.0, 1200 / max(height, width))

    if scale < 1:
        preview = cv2.resize(
            preview, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    print(f"\nDraw a box around the full card: {filename}")
    print("Press ENTER or SPACE to accept; ESC uses the full image.")

    x, y, w, h = cv2.selectROI(
        "Select card area",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select card area")

    if w == 0 or h == 0:
        return image

    x, y, w, h = [int(value / scale) for value in (x, y, w, h)]
    return image[y:y + h, x:x + w]


def make_versions(image):
    """Make several image versions to handle different lighting and print."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    if width < 1600:
        scale = 1600 / width
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    contrast = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(gray)

    sharpened = cv2.addWeighted(
        contrast,
        1.5,
        cv2.GaussianBlur(contrast, (0, 0), 2),
        -0.5,
        0,
    )

    _, thresholded = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return [gray, contrast, sharpened, thresholded]


def run_ocr(image):
    """OCR both the full card and its student-details area."""
    height, width = image.shape[:2]

    # These proportions match the supplied cards' right-side detail area.
    details = image[
        int(height * 0.38):int(height * 0.76),
        int(width * 0.40):int(width * 0.97),
    ]

    regions = [image]
    if details.size:
        regions.append(details)

    results = []
    for region in regions:
        for version in make_versions(region):
            for psm in (6, 11):
                text = pytesseract.image_to_string(
                    version,
                    lang="eng",
                    config=f"--oem 3 --psm {psm}",
                )
                results.append(text)

    return results


def clean_line(line):
    line = line.replace("|", "I")
    return re.sub(r"\s+", " ", line).strip(" \t:;|")


def extract_fields(text):
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    whole_text = "\n".join(lines)

    fields = {
        "name": "",
        "admission_no": "",
        "course": "",
        "address": "",
        "gender": "",
        "mobile": "",
        "date_of_birth": "",
        "blood_group": "",
    }

    # Admission number and the name printed just above it.
    admission_index = None
    for i, line in enumerate(lines):
        match = re.search(
            r"admission\s*(?:no\.?|number)?\s*[:#.]?\s*([A-Z0-9/-]+)",
            line,
            re.IGNORECASE,
        )
        if match:
            fields["admission_no"] = match.group(1).strip()
            admission_index = i
            break

    if admission_index is not None:
        for line in reversed(lines[max(0, admission_index - 3):admission_index]):
            candidate = re.sub(r"[^A-Za-z .'-]", " ", line)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if (
                len(candidate) >= 3
                and not re.search(
                    r"college|engineering|identity|address|valid|mobile|blood",
                    candidate,
                    re.IGNORECASE,
                )
            ):
                fields["name"] = candidate
                break

        # Course may wrap onto the following OCR line.
        for i in range(admission_index + 1, len(lines)):
            if re.search(r"\bB\s*\.?\s*Tech\b", lines[i], re.IGNORECASE):
                course = lines[i]
                if i + 1 < len(lines) and re.search(
                    r"artificial|intelligence|machine learning",
                    lines[i + 1],
                    re.IGNORECASE,
                ):
                    course += " " + lines[i + 1]
                fields["course"] = course
                break

    # Address can wrap across several lines.
    for i, line in enumerate(lines):
        if re.search(r"\badd(?:ress)?\s*:", line, re.IGNORECASE):
            first = re.sub(
                r"^.*?\badd(?:ress)?\s*:\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            parts = [first] if first else []

            for following in lines[i + 1:]:
                if re.search(
                    r"\bm\s*/\s*f\b|mobile|d\s*\.?\s*o\s*\.?\s*b|blood\s*group",
                    following,
                    re.IGNORECASE,
                ):
                    break
                if re.search(
                    r"\badmission\b|\bb\s*\.?\s*tech\b",
                    following,
                    re.IGNORECASE,
                ):
                    break
                parts.append(following)

            fields["address"] = " ".join(parts).strip()
            break

    # Phone number: prefer a line labeled Mobile.
    for line in lines:
        if "mobile" in line.lower():
            match = re.search(r"(\+?\d[\d\s()-]{7,}\d)", line)
            if match:
                digits = re.sub(r"\D", "", match.group(1))
                if len(digits) >= 10:
                    fields["mobile"] = digits[-10:]
                    break

    # Fallback for a missed Mobile label. Use the last 10-digit Indian
    # mobile-looking number, since the college number appears earlier.
    if not fields["mobile"]:
        numbers = re.findall(
            r"(?<!\d)[6-9](?:[\s-]*\d){9}(?!\d)",
            whole_text,
        )
        if numbers:
            fields["mobile"] = re.sub(r"\D", "", numbers[-1])

    # DOB: first search the DOB line, then search all OCR text.
    date_pattern = (
        r"([0-9Oo]{1,2}\s*[-/.]\s*[0-9Oo]{1,2}"
        r"\s*[-/.]\s*[0-9Oo]{4})"
    )

    for line in lines:
        if re.search(r"d\s*\.?\s*o\s*\.?\s*b|dob", line, re.IGNORECASE):
            match = re.search(date_pattern, line, re.IGNORECASE)
            if match:
                fields["date_of_birth"] = re.sub(
                    r"\s+", "", match.group(1)
                ).replace("O", "0").replace("o", "0")
                break

    if not fields["date_of_birth"]:
        match = re.search(date_pattern, whole_text, re.IGNORECASE)
        if match:
            fields["date_of_birth"] = re.sub(
                r"\s+", "", match.group(1)
            ).replace("O", "0").replace("o", "0")

    # Read gender only as the value after the M/F label.
    for line in lines:
        if re.search(r"\bm\s*/\s*f\b", line, re.IGNORECASE):
            match = re.search(r":\s*([MF])\b", line, re.IGNORECASE)
            if match:
                fields["gender"] = match.group(1).upper()
                break

    # Blood group, e.g. A+ or O+.
    for line in lines:
        if "blood" in line.lower():
            match = re.search(r"\b(AB|A|B|O)\s*([+-])", line, re.IGNORECASE)
            if match:
                fields["blood_group"] = (
                    match.group(1).upper() + match.group(2)
                )
                break

    return fields


def merge_fields(parsed_candidates):
    """Vote field-by-field across OCR passes, rather than choosing one pass."""
    merged = {}

    field_names = (
        "name",
        "admission_no",
        "course",
        "address",
        "gender",
        "mobile",
        "date_of_birth",
        "blood_group",
    )

    for field in field_names:
        values = [
            row[field]
            for row in parsed_candidates
            if row.get(field)
        ]

        if not values:
            merged[field] = ""
            continue

        groups = {}
        for value in values:
            key = re.sub(r"[^a-z0-9]", "", value.lower())
            groups.setdefault(key, []).append(value)

        winning_group = max(
            groups.values(),
            key=lambda group: (len(group), max(map(len, group))),
        )
        merged[field] = max(winning_group, key=len)

    return merged


def score_fields(fields):
    return sum(bool(value) for value in fields.values())


def main():
    if not configure_tesseract():
        return

    image_paths = choose_images()
    if not image_paths:
        print("No images selected.")
        return

    output_dir = Path(image_paths[0]).parent / "id_card_results"
    output_dir.mkdir(exist_ok=True)

    rows = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not open image: {image_path}")
            continue

        card = crop_card(image, Path(image_path).name)

        try:
            ocr_texts = run_ocr(card)
        except pytesseract.TesseractNotFoundError:
            print("Tesseract could not be started. Check TESSERACT_PATH.")
            return

        parsed_candidates = [extract_fields(text) for text in ocr_texts]
        fields = merge_fields(parsed_candidates)

        # Save the OCR text pass that recognized the most fields for review.
        best_text = max(
            ocr_texts,
            key=lambda text: score_fields(extract_fields(text)),
            default="",
        )

        fields["image"] = Path(image_path).name
        rows.append(fields)

        print(f"\n--- {fields['image']} ---")
        for key, value in fields.items():
            if key != "image":
                print(f"{key}: {value or 'Not detected'}")

        text_path = output_dir / f"{Path(image_path).stem}_ocr.txt"
        text_path.write_text(best_text, encoding="utf-8")

    columns = [
        "image",
        "name",
        "admission_no",
        "course",
        "address",
        "gender",
        "mobile",
        "date_of_birth",
        "blood_group",
    ]

    csv_path = output_dir / "students.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStudent details saved to: {csv_path}")
    print(f"Full OCR text saved in: {output_dir}")


if __name__ == "__main__":
    main()