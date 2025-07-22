import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import os


def create_qa_pdf(csv_file_path, output_pdf_path):
    """
    Create a PDF with Q&A format from CSV data

    Args:
        csv_file_path (str): Path to the input CSV file
        output_pdf_path (str): Path where the PDF will be saved
    """

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Validate required columns
    required_columns = ["Review Text", "Developer Reply Text"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        print(f"Found columns: {list(df.columns)}")
        return

    # Create PDF document
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Get sample stylesheet and create custom styles
    styles = getSampleStyleSheet()

    # Custom styles for Q&A
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=HexColor("#2E4A99"),
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=20,
        leftIndent=20,
        textColor=HexColor("#1F4E79"),
        fontName="Helvetica-Bold",
    )

    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=15,
        leftIndent=40,
        rightIndent=20,
        textColor=HexColor("#333333"),
        fontName="Helvetica",
    )

    # Build the story (content) for the PDF
    story = []

    # Add title
    story.append(Paragraph("Game Reviews Q&A", title_style))
    story.append(Spacer(1, 20))

    # Process each row in the CSV
    for index, row in df.iterrows():
        review_text = str(row["Review Text"]).strip()
        developer_reply = str(row["Developer Reply Text"]).strip()

        # Skip empty rows
        if (
            not review_text
            or review_text == "nan"
            or not developer_reply
            or developer_reply == "nan"
        ):
            continue

        # Add question number and text
        question_number = index + 1
        question = f"Q{question_number}: {review_text}"
        story.append(Paragraph(question, question_style))

        # Add answer
        answer = f"A: {developer_reply}"
        story.append(Paragraph(answer, answer_style))

        # Add some space between Q&A pairs
        story.append(Spacer(1, 10))

    # Build PDF
    try:
        doc.build(story)
        print(f"PDF successfully created: {output_pdf_path}")
        print(
            f"Total Q&A pairs processed: {len([row for _, row in df.iterrows() if str(row['Review Text']).strip() and str(row['Developer Reply Text']).strip()])}"
        )
    except Exception as e:
        print(f"Error creating PDF: {e}")


def main():
    # Configuration
    csv_file = "filtered_reviews.csv"  # Input CSV file
    pdf_file = "game_reviews_qa.pdf"  # Output PDF file

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found in current directory")
        print("Please make sure the CSV file is in the same folder as this script")
        return

    print("Starting CSV to PDF conversion...")
    print(f"Input file: {csv_file}")
    print(f"Output file: {pdf_file}")

    # Create the PDF
    create_qa_pdf(csv_file, pdf_file)


if __name__ == "__main__":
    main()
