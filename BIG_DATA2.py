import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def generate_messy_data(filename: str = "zamowienia_messy.csv", n: int = 500) -> None:
    np.random.seed(42)

    klienci = [
        "Anna Kowalska", "  Jan Nowak", "Anna Kowalska", "PIOTR WIŚNIEWSKI",
        "katarzyna lewandowska", "Tomasz Zieliński ", "Marta Wójcik",
        "anna kowalska ", "Krzysztof Kamiński", " Magdalena Dąbrowska"
    ]
    produkty = [
        "Laptop", "Mysz", "Klawiatura", "Monitor", "laptop", "MYSZ",
        "Słuchawki", "Pendrive", "monitor", "Webcam"
    ]
    kategorie = [
        "Elektronika", "elektronika", "ELEKTRONIKA",
        "Akcesoria", "akcesoria", "Akcesoria "
    ]
    miasta = [
        "Warszawa", "Kraków", "warszawa", "Gdańsk", "WROCŁAW",
        "Poznań", "Łódź ", " Warszawa", "kraków"
    ]

    start_date = datetime(2025, 1, 1)
    daty_iso = [
        (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in np.random.randint(0, 300, n // 2)
    ]
    daty_pl = [
        (start_date + timedelta(days=int(d))).strftime("%d.%m.%Y")
        for d in np.random.randint(0, 300, n // 2)
    ]
    daty = daty_iso + daty_pl
    np.random.shuffle(daty)

    df = pd.DataFrame({
        "order_id": range(1001, 1001 + n),
        "klient": np.random.choice(klienci, n),
        "produkt": np.random.choice(produkty, n),
        "kategoria": np.random.choice(kategorie, n),
        "miasto": np.random.choice(miasta, n),
        "ilosc": np.random.choice(
            [1, 2, 3, 5, -1, 0],
            n,
            p=[0.5, 0.2, 0.15, 0.1, 0.025, 0.025]
        ),
        "cena_jednostkowa": np.random.choice(
            ["199.99", "299,99", "1 499.00", "89.50", "2999", "399.00 zł", None, "abc"],
            n
        ),
        "data_zamowienia": daty,
        "email": np.random.choice(
            [
                "anna@gmail.com", "JAN@WP.PL", "piotr.w@onet", "marta@gmail.com",
                "tomasz@interia.pl", None, "krzysztof.k@gmail.com", "brak"
            ],
            n
        )
    })

    for col in ["miasto", "kategoria", "data_zamowienia"]:
        df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

    df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)

    df.to_csv(filename, index=False)
    print(f"Wygenerowano plik '{filename}' — {len(df)} wierszy")


def load_data(filename: str = "zamowienia_messy.csv") -> pd.DataFrame:
    df = pd.read_csv(filename)
    print("Wczytano dane:", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    for col in ["klient", "produkt", "kategoria", "miasto"]:
        df[col] = df[col].astype(str).str.strip()

    df["klient"] = df["klient"].str.lower().str.title()
    df["miasto"] = df["miasto"].str.lower().str.title()
    df["produkt"] = df["produkt"].str.lower()
    df["kategoria"] = df["kategoria"].str.lower()

    df["data_zamowienia"] = pd.to_datetime(
        df["data_zamowienia"],
        errors="coerce",
        dayfirst=True
    )

    c = df["cena_jednostkowa"].astype(str)
    c = c.str.replace(",", ".", regex=False)
    c = c.str.replace(" zł", "", regex=False)
    c = c.str.replace(" ", "", regex=False)
    df["cena_jednostkowa"] = pd.to_numeric(c, errors="coerce")

    df = df.dropna(subset=["cena_jednostkowa", "data_zamowienia"])

    df["miasto"] = df["miasto"].fillna("unknown")
    df["kategoria"] = df["kategoria"].fillna("unknown")

    df["email"] = df["email"].fillna("brak_emaila")
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df.loc[df["email"].isin(["", "nan", "brak"]), "email"] = "brak_emaila"

    df = df[df["ilosc"] > 0]

    return df


def add_transformations(df: pd.DataFrame) -> pd.DataFrame:
    df["wartosc_zamowienia"] = df["ilosc"] * df["cena_jednostkowa"]
    df["rok"] = df["data_zamowienia"].dt.year
    df["miesiac"] = df["data_zamowienia"].dt.month
    df["nazwa_dnia"] = df["data_zamowienia"].dt.day_name()

    pattern = re.compile(r"^[^@]+@[^@]+\.[^@]+$")

    def is_valid_email(x: str) -> bool:
        if x == "brak_emaila":
            return False
        return bool(pattern.match(str(x)))

    df["email_poprawny"] = df["email"].apply(is_valid_email)
    return df


def analyses(df: pd.DataFrame):
    print("\n=== Łączna wartość zamówień w każdym miesiącu ===")
    miesiecznie = (
        df.groupby(["rok", "miesiac"])["wartosc_zamowienia"]
          .sum()
          .reset_index()
          .sort_values(["rok", "miesiac"])
    )
    print(miesiecznie)

    print("\n=== Top 5 klientów po łącznej wartości zamówień ===")
    top_klienci = (
        df.groupby("klient")["wartosc_zamowienia"]
          .sum()
          .sort_values(ascending=False)
          .head(5)
    )
    print(top_klienci)

    print("\n=== Średnia wartość zamówienia w każdej kategorii ===")
    srednia_kategoria = (
        df.groupby("kategoria")["wartosc_zamowienia"]
          .mean()
          .sort_values(ascending=False)
    )
    print(srednia_kategoria)

    return miesiecznie


def plot_monthly_bar(miesiecznie: pd.DataFrame) -> None:
    miesiecznie = miesiecznie.copy()
    miesiecznie["rok_miesiac"] = (
        miesiecznie["rok"].astype(str)
        + "-"
        + miesiecznie["miesiac"].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(12, 6))
    plt.bar(miesiecznie["rok_miesiac"], miesiecznie["wartosc_zamowienia"])
    plt.xticks(rotation=45)
    plt.xlabel("Miesiąc")
    plt.ylabel("Łączna wartość zamówień")
    plt.title("Łączna wartość zamówień w każdym miesiącu")
    plt.tight_layout()
    plt.show()


def save_clean(df: pd.DataFrame, filename: str = "zamowienia_clean.csv") -> None:
    df.to_csv(filename, index=False)
    print(f"Zapisano oczyszczone dane do '{filename}'")


def main():
    generate_messy_data("zamowienia_messy.csv", n=500)
    df = load_data("zamowienia_messy.csv")
    df = clean_data(df)
    df = add_transformations(df)
    miesiecznie = analyses(df)
    plot_monthly_bar(miesiecznie)
    save_clean(df, "zamowienia_clean.csv")

# cos mi nie chce odpalic main
if __name__ == "__main__":
    main()