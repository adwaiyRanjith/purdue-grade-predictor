import pandas as pd
import numpy as np


# function to clean sum16-sum21, spring 2022, and summer 2022
def clean1(fin_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(fin_path, sheet_name=sheet_name, header=None)
    
    grade_letters = df.iloc[7, 9:].tolist()
    grade_letters = [g for g in grade_letters if pd.notna(g)]
    
    df.columns = df.iloc[8]
    df = df.iloc[9:].reset_index(drop=True)
    
    new_columns = list(df.columns[:9]) + grade_letters
    df = df.iloc[:, :len(new_columns)]
    df.columns = new_columns
    

    df['Subject'] = df['Subject'].ffill()
    df['Subject Desc'] = df['Subject Desc'].ffill()
    df['Course Number'] = df['Course Number'].ffill()
    df['Title'] = df['Title'].ffill()
    df['Academic Period'] = df['Academic Period'].ffill()
    df['Academic Period Desc'] = df['Academic Period Desc'].ffill()
    keep = ['Subject', 'Subject Desc', 'Course Number', 'Title',
        'Academic Period', 'Academic Period Desc', 'Section', 'CRN', 'Instructor',
        'A', 'A-', 'A+', 'B', 'B-', 'B+', 'C', 'C-', 'C+', 'D', 'D-', 'D+', 'F', 'W']
    df = df[[col for col in keep if col in df.columns]]

    df = df.dropna(subset=['Instructor', 'Section', 'CRN'])
    return df

# function to clean fall 2022
def clean2(fin_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(fin_path, sheet_name=sheet_name, header=None)
    
    grade_letters = df.iloc[7, 9:].tolist()
    grade_letters = [g for g in grade_letters if pd.notna(g)]
    
    df.columns = df.iloc[8]
    df = df.iloc[9:].reset_index(drop=True)
    identifier_cols = df.iloc[:, :9]
    grade_cols = df.iloc[:, 10::2].iloc[:, :len(grade_letters)]

    df = pd.concat([identifier_cols, grade_cols], axis=1)
    df.columns = list(df.columns[:9]) + grade_letters
    df['Subject'] = df['Subject'].ffill()
    df['Subject Desc'] = df['Subject Desc'].ffill()
    df['Course Number'] = df['Course Number'].ffill()
    df['Title'] = df['Title'].ffill()
    df['Academic Period'] = df['Academic Period'].ffill()
    df['Academic Period Desc'] = df['Academic Period Desc'].ffill()
    keep = ['Subject', 'Subject Desc', 'Course Number', 'Title',
        'Academic Period', 'Academic Period Desc', 'Section', 'CRN', 'Instructor',
        'A', 'A-', 'A+', 'B', 'B-', 'B+', 'C', 'C-', 'C+', 'D', 'D-', 'D+', 'F', 'W']
    df = df[[col for col in keep if col in df.columns]]

    df = df.dropna(subset=['Instructor', 'Section', 'CRN'])
    return df
    

# function to clean spring 2023
def clean3(fin_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(fin_path, sheet_name=sheet_name, header=None)
    
    df.columns = df.iloc[7]
    df = df.iloc[8:].reset_index(drop=True)
    
    df['Subject'] = df['Subject'].ffill()
    df['Subject Desc'] = df['Subject Desc'].ffill()
    df['Course Number'] = df['Course Number'].ffill()
    df['Title'] = df['Title'].ffill()
    df['Academic Period'] = df['Academic Period'].ffill()
    df['Academic Period Desc'] = df['Academic Period Desc'].ffill()
    
    keep = ['Subject', 'Subject Desc', 'Course Number', 'Title',
        'Academic Period', 'Academic Period Desc', 'Section', 'CRN', 'Instructor',
        'A', 'A-', 'A+', 'B', 'B-', 'B+', 'C', 'C-', 'C+', 'D', 'D-', 'D+', 'F', 'W']
    df = df[[col for col in keep if col in df.columns]]
    df = df.dropna(subset=['Instructor', 'Section', 'CRN'])
    return df


if __name__ == "__main__":
    for sheet in ["Sum16-Sum21", "Spring 2022", "Summer 2022"]:
        df = clean1("data/raw/grades.xlsx", sheet)
        
    

    for sheet in ["Fall 2022"]:
        df = clean2("data/raw/grades.xlsx", sheet)
        
    
    for sheet in ["Spring 2023"]:
        df = clean3("data/raw/grades.xlsx", sheet)
        
