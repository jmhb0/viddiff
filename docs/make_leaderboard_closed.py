import pandas as pd

# File paths
input_csv = "docs/static/leaderboard/data/web-csv-closed.csv"
output_html = "docs/static/leaderboard/html/web-csv-closed.html"

# Read CSV
df = pd.read_csv(input_csv)

# Ensure necessary columns exist
if "model_name" not in df.columns or "model_link" not in df.columns:
    raise ValueError("CSV file must contain 'model_name' and 'model_link' columns.")

# Create a new "Model" column with hyperlinks
df["Model"] = df.apply(lambda row: f'<a href="{row["model_link"]}" target="_blank">{row["model_name"]}</a>', axis=1)

# Identify "yes/no" indicator columns (assumed to be the last 4 columns)
num_columns = len(df.columns)
indicator_columns = df.columns[-4:]  # The last 4 columns are assumed to be the indicators
data_columns = df.columns[-8:-4]  # The 4 columns before the indicators are the data columns

# Drop the "yes/no" indicator columns from the final table
df = df.drop(columns=["model_name", "model_link"] + list(indicator_columns))

# Reorder columns so "Model" is first
columns_order = ["Model"] + [col for col in df.columns if col != "Model"]
df = df[columns_order]

# Generate HTML rows with yellow highlights for "yes" values
def generate_row(row):
    row_html = []
    for col in df.columns:
        if col in data_columns:  # Check if it's a data column
            indicator_col = indicator_columns[data_columns.get_loc(col)]
            if row[indicator_col].strip().lower() == "yes":  # Apply yellow highlight
                row_html.append(f'<td style="background-color: #fff7cc;">{row[col]}</td>')
            else:
                row_html.append(f'<td>{row[col]}</td>')
        else:
            row_html.append(f'<td>{row[col]}</td>')
    return "<tr>" + "".join(row_html) + "</tr>"

# Generate HTML
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaderboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
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

<h2>Leaderboard</h2>
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

