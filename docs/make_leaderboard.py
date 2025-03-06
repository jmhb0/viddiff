import pandas as pd

# File paths
datasets = [
    ("docs/static/leaderboard/data/web-csv-open.csv", "docs/static/leaderboard/html/web-csv-open.html", "Leaderboard - Open"),
    ("docs/static/leaderboard/data/web-csv-closed.csv", "docs/static/leaderboard/html/web-csv-closed.html", "Leaderboard - Closed"),
]

for input_csv, output_html, page_title in datasets:
    # Read CSV
    df = pd.read_csv(input_csv)

    # Ensure necessary columns exist
    if "model_name" not in df.columns or "model_link" not in df.columns:
        raise ValueError(f"CSV file {input_csv} must contain 'model_name' and 'model_link' columns.")

    # **Fix: Add "Model" before dropping `model_name` and `model_link`**
    df.insert(0, "Model", df.apply(lambda row: f'<a href="{row["model_link"]}" target="_blank">{row["model_name"]}</a>', axis=1))

    # Drop unnecessary columns
    df = df.drop(columns=["model_name", "model_link"])

    # **Special logic for web-csv-closed.csv**
    data_columns = []
    statsig_columns = []
    column_map = {}  # Maps data columns to their corresponding statsig columns

    if "closed" in input_csv:
        # **Find all "statsig" columns (these contain yes/no values)**
        statsig_columns = [col for col in df.columns if col.startswith("statsig")]

        # **Find the corresponding data columns (these come before the statsig columns)**
        num_statsig = len(statsig_columns)
        data_columns = df.columns[:num_statsig]  # First N columns match the last N statsig columns

        # **Create a mapping of data column → corresponding statsig column**
        for i in range(num_statsig):
            column_map[data_columns[i]] = statsig_columns[i]

    # **Generate HTML rows with yellow highlights before dropping statsig columns**
    def generate_row(row):
        row_html = []
        for col in df.columns:
            if "closed" in input_csv and col in column_map:  # Ensure valid data column
                statsig_col = column_map[col]  # Get corresponding statsig column
                if statsig_col in row and str(row[statsig_col]).strip().lower() == "yes":
                    row_html.append(f'<td style="background-color: #fff7cc;">{row[col]}</td>')
                else:
                    row_html.append(f'<td>{row[col]}</td>')
            else:
                row_html.append(f'<td>{row[col]}</td>')
        return "<tr>" + "".join(row_html) + "</tr>"

    # **Now drop "statsig" columns AFTER generating rows**
    if "closed" in input_csv:
        df = df.drop(columns=statsig_columns)

    # **Ensure "Model" is first**
    df = df[["Model"] + [col for col in df.columns if col != "Model"]]

    # Generate HTML
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{page_title}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: white;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background-color: white;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #f2f2f2;
                cursor: pointer;
            }}
            th:hover {{
                background-color: #ddd;
            }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }} /* Alternating row colors */
            tr:nth-child(odd)  {{ background-color: #ffffff; }}
            a {{
                text-decoration: none;
                color: #007bff;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>

    <h4>{page_title}</h4>
    <table id="leaderboard" class="display">
        <thead>
            <tr>
                {" ".join(f"<th>{col}</th>" for col in df.columns)}
            </tr>
        </thead>
        <tbody>
            {" ".join(generate_row(row) for _, row in df.iterrows())}
        </tbody>
    </table>

    <script>
        $(document).ready( function () {{
            $('#leaderboard').DataTable({{
                "order": [[ {df.columns.get_loc("Average")}, "desc" ]], // Sort by "Average" column descending
                "searching": false,  // Disable search bar
                "paging": false,  // Disable pagination (removes "Show X entries")
                "info": false,  // Removes "Showing X of Y entries" text
                "columnDefs": [
                    {{ "type": "html", "targets": [0] }}  // Enable sorting for hyperlinks in "Model"
                ]
            }});
        }});
    </script>

    </body>
    </html>
    """

    # Save to file
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Leaderboard generated: {output_html}")

