from data_cleaner import clean1, clean2, clean3
import pathlib
import pandas as pd

def load_excel_data() -> pd.DataFrame:
    path = "data/raw/grades.xlsx"
    sheets = []

    for sheet in ["Sum16-Sum21", "Spring 2022", "Summer 2022"]:
        sheets.append(clean1(path, sheet))
    
    sheets.append(clean2(path, "Fall 2022"))
    
    for sheet in ["Spring 2023", "Fall 2023"]:
        sheets.append(clean3(path, sheet))
    
    df = pd.concat(sheets, ignore_index=True)
    return df


grade_cols = ['A', 'A-', 'A+', 'B', 'B-', 'B+', 'C', 'C-', 'C+', 'D', 'D-', 'D+', 'F', 'W']

CSV_URLS = [
    "https://github.com/eduxstad/boiler-grades/raw/main/fall2021.csv",
    "https://github.com/eduxstad/boiler-grades/raw/main/summer2023.csv",
    "https://github.com/eduxstad/boiler-grades/raw/main/spring2024.csv",
    "https://github.com/eduxstad/boiler-grades/raw/main/fall2024.csv",
    "https://github.com/eduxstad/boiler-grades/raw/main/spring2025.csv",
    "https://github.com/eduxstad/boiler-grades/raw/main/fall2025.csv",
]

def load_csvs() -> pd.DataFrame:
    keep = ['Subject', 'Subject Desc', 'Course Number', 'Title',
            'Academic Period', 'Academic Period Desc', 'Section', 'CRN', 'Instructor',
            'A', 'A-', 'A+', 'B', 'B-', 'B+', 'C', 'C-', 'C+', 'D', 'D-', 'D+', 'F', 'W']
    
    sheets = []
    for url in CSV_URLS:
        df = pd.read_csv(url, sep=None, engine='python', on_bad_lines='skip')
        # rename w_f to WF if present
        df = df.rename(columns={'w_f': 'WF'})
        # ffill merged cells
        df['Subject'] = df['Subject'].ffill()
        df['Subject Desc'] = df['Subject Desc'].ffill()
        df['Course Number'] = df['Course Number'].ffill()
        df['Title'] = df['Title'].ffill()
        df['Academic Period'] = df['Academic Period'].ffill()
        df['Academic Period Desc'] = df['Academic Period Desc'].ffill()
        # keep only columns we want
        df = df[[col for col in keep if col in df.columns]]
        df = df.dropna(subset=['Instructor', 'Section', 'CRN'])
        sheets.append(df)
    
    return pd.concat(sheets, ignore_index=True)

def load_all_data() -> pd.DataFrame:
    excel_df = load_excel_data()
    csv_df = load_csvs()
    df = pd.concat([excel_df, csv_df], ignore_index=True)
    
    for col in grade_cols:
        df[col] = df[col].astype(str).str.replace('%', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].where(df[col] <= 1, df[col] / 100)
    
    return df


if __name__ == "__main__":
    df = load_all_data()
    print(df[grade_cols].describe())
    print(df.tail(5).to_string())