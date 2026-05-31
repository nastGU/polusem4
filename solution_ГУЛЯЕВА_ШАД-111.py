# -*- coding: utf-8 -*-

# %%

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# КОНФИГУРАЦИЯ (единственный параметр – минимальный размер группы)
DATA_PATH = r"C:\Users\nasty\Downloads\data_train\data_train"   # измените под свой путь, где лежат файлы с данными
OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MIN_RESPONDENTS_PER_GROUP = 20          # минимальное число респондентов для анализа (защита от малых выборок)
WEIGHT_AGG = "max"                      # агрегация веса при дубликатах

# ДРУГИЕ ФУНКЦИИ
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def find_gap_threshold(values):
    """
    Порог = последнее значение перед максимальным разрывом.
    Если выборка слишком мала или все значения равны -> None.
    """
    if len(values) < MIN_RESPONDENTS_PER_GROUP:
        return None
    vals = np.sort(values)
    if vals[0] == vals[-1]:
        return None
    gaps = np.diff(vals)
    max_gap_idx = np.argmax(gaps)
    return vals[max_gap_idx]   # всё, что строго больше – аномалия

# ЗАГРУЗКА ДАННЫХ
def load_data():
    pattern = os.path.join(DATA_PATH, "month=*", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"Нет файлов по пути {pattern}")
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Загружено строк: {len(df):,}")
    return df

# ФИЛЬТРАЦИЯ И ПОДГОТОВКА
def prepare_data(df):
    df = df.copy()
    # Оставляем только строки с BrandinDelivery == 1 и непустой категорией
    mask = (df["BrandinDelivery"] == 1.0) & (df["CategoryNameDelivery"].notna())
    df = df[mask]
    df["researchdate"] = pd.to_datetime(df["researchdate"])
    df["month"] = df["researchdate"].dt.to_period("M").astype(str)
    df["ots"] = df["Weight"]   # дневной вес для графиков
    print(f"После фильтрации строк: {len(df):,}")
    return df

# АГРЕГАЦИЯ DAILY_OTS
def build_daily_ots(df):
    group_cols = [
        "SubjectID", "researchdate", "month",
        "BrandID", "Brand", "CategoryNameDelivery"
    ]
    daily = df.groupby(group_cols).agg(
        cnt=("QueryText", "size"),
        weight=("Weight", WEIGHT_AGG)
    ).reset_index()
    daily["daily_ots"] = daily["weight"] * daily["cnt"]
    return daily

# ОБНАРУЖЕНИЕ АНОМАЛИЙ 
def detect_anomalies_gap(daily):
    groups = daily.groupby(["CategoryNameDelivery", "BrandID", "researchdate"])
    anomaly_records = []

    for (cat, brand, date), group in groups:
        values = group["daily_ots"].values
        threshold = find_gap_threshold(values)
        if threshold is None:
            continue
        anomaly_mask = group["daily_ots"] > threshold
        if not anomaly_mask.any():
            continue
        for _, row in group[anomaly_mask].iterrows():
            score = row["daily_ots"] / threshold
            reason = (f"daily_ots={row['daily_ots']:.1f} > порог={threshold:.1f} "
                      f"(максимальный разрыв в отсортированном ряду), score={score:.2f}")
            anomaly_records.append({
                "SubjectID": row["SubjectID"],
                "researchdate": row["researchdate"],
                "BrandID": row["BrandID"],
                "Brand": row["Brand"],
                "CategoryNameDelivery": cat,
                "daily_ots": row["daily_ots"],
                "score": score,
                "threshold": threshold,
                "reason": reason
            })
    anomalies = pd.DataFrame(anomaly_records) if anomaly_records else pd.DataFrame()
    print(f"Найдено аномальных триггеров: {len(anomalies)}")
    return anomalies

# СОХРАНЕНИЕ ФАЙЛОВ
def save_anomalies(anomalies):
    if anomalies.empty:
        pairs = pd.DataFrame(columns=["SubjectID", "researchdate"])
    else:
        pairs = anomalies[["SubjectID", "researchdate"]].drop_duplicates()
    pairs.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)
    print(f"Сохранено {len(pairs)} пар (SubjectID, researchdate)")
    return pairs

def save_anomaly_reasons(anomalies):
    if anomalies.empty:
        empty = pd.DataFrame(columns=[
            "SubjectID", "researchdate", "BrandID", "Brand",
            "CategoryNameDelivery", "daily_ots", "score", "threshold", "reason"
        ])
        empty.to_csv(os.path.join(OUTPUT_DIR, "anomaly_reasons.csv"), index=False)
    else:
        anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomaly_reasons.csv"), index=False)
    print("Сохранён anomaly_reasons.csv")

# УДАЛЕНИЕ АНОМАЛЬНЫХ ДНЕЙ
def remove_anomalies(df, anomaly_pairs):
    if anomaly_pairs.empty:
        return df.copy()
    anomaly_set = set(zip(anomaly_pairs["SubjectID"], anomaly_pairs["researchdate"]))
    mask = df.apply(lambda row: (row["SubjectID"], row["researchdate"]) in anomaly_set, axis=1)
    cleaned = df[~mask].copy()
    print(f"До удаления: {len(df)} строк, после: {len(cleaned)} строк")
    return cleaned

# ОБЯЗАТЕЛЬНЫЕ ГРАФИКИ (п.8.1)
def plot_total_ots_before_after(before_df, after_df):
    before = before_df.groupby("researchdate")["ots"].sum().reset_index(name="before")
    after = after_df.groupby("researchdate")["ots"].sum().reset_index(name="after")
    merged = before.merge(after, on="researchdate", how="outer").fillna(0)
    plt.figure(figsize=(16,6))
    plt.plot(merged["researchdate"], merged["before"], label="Before")
    plt.plot(merged["researchdate"], merged["after"], label="After")
    plt.legend(); plt.xticks(rotation=45)
    plt.title("Total OTS Before/After Cleaning")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "total_ots_before_after.png"))
    plt.close()

def plot_category_ots_change(before_df, after_df, cat_col="CategoryNameDelivery"):
    before = before_df.groupby(cat_col)["ots"].sum().reset_index(name="before")
    after = after_df.groupby(cat_col)["ots"].sum().reset_index(name="after")
    merged = before.merge(after, on=cat_col, how="outer").fillna(0)
    merged["change_pct"] = 100 * (merged["after"] - merged["before"]) / (merged["before"] + 1e-9)
    merged = merged.sort_values("change_pct")
    plt.figure(figsize=(16,8))
    plt.barh(merged[cat_col].astype(str), merged["change_pct"])
    plt.title("Category OTS Change (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "category_ots_change.png"))
    plt.close()

def plot_daily_anomaly_count(anomaly_pairs):
    if anomaly_pairs.empty:
        plt.figure(); plt.text(0.5,0.5,"No anomalies"); plt.savefig(os.path.join(PLOTS_DIR,"daily_anomaly_count.png")); plt.close()
        return
    counts = anomaly_pairs.groupby("researchdate").size().reset_index(name="count")
    plt.figure(figsize=(16,6))
    plt.bar(counts["researchdate"].astype(str), counts["count"])
    plt.xticks(rotation=45)
    plt.title("Daily Anomaly Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "daily_anomaly_count.png"))
    plt.close()

# ДОПОЛНИТЕЛЬНЫЕ АНАЛИТИЧЕСКИЕ ГРАФИКИ (п.8.2)
def plot_change_by_demographic(before_df, after_df, demo_col, output_dir):
    before = before_df.groupby(demo_col)["ots"].sum().reset_index(name="before")
    after = after_df.groupby(demo_col)["ots"].sum().reset_index(name="after")
    merged = before.merge(after, on=demo_col, how="outer").fillna(0)
    merged["change_pct"] = 100 * (merged["after"] - merged["before"]) / (merged["before"] + 1e-9)
    merged = merged.sort_values("change_pct")
    plt.figure(figsize=(12, 8))
    plt.barh(merged[demo_col].astype(str), merged["change_pct"])
    plt.xlabel("Изменение OTS (%)")
    plt.title(f"Изменение OTS после очистки по признаку '{demo_col}'")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"change_by_{demo_col}.png"))
    plt.close()

def plot_change_by_resource_feature(before_df, after_df, feature_col, output_dir):
    before = before_df.groupby(feature_col)["ots"].sum().reset_index(name="before")
    after = after_df.groupby(feature_col)["ots"].sum().reset_index(name="after")
    merged = before.merge(after, on=feature_col, how="outer").fillna(0)
    merged["change_pct"] = 100 * (merged["after"] - merged["before"]) / (merged["before"] + 1e-9)
    merged = merged.sort_values("change_pct")
    plt.figure(figsize=(12, 8))
    plt.barh(merged[feature_col].astype(str), merged["change_pct"])
    plt.xlabel("Изменение OTS (%)")
    plt.title(f"Изменение OTS после очистки по признаку '{feature_col}'")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"change_by_{feature_col}.png"))
    plt.close()

def plot_brand_ots_trend(before_df, after_df, brand_id, cat_delivery, output_dir):
    before_brand = before_df[(before_df["BrandID"] == brand_id) & 
                             (before_df["CategoryNameDelivery"] == cat_delivery)]
    after_brand = after_df[(after_df["BrandID"] == brand_id) & 
                           (after_df["CategoryNameDelivery"] == cat_delivery)]
    if before_brand.empty and after_brand.empty:
        print(f"Бренд {brand_id} в категории {cat_delivery} не найден.")
        return
    before_day = before_brand.groupby("researchdate")["ots"].sum().reset_index()
    after_day = after_brand.groupby("researchdate")["ots"].sum().reset_index()
    merged = before_day.merge(after_day, on="researchdate", how="outer", suffixes=("_before", "_after")).fillna(0)
    plt.figure(figsize=(12, 6))
    plt.plot(merged["researchdate"], merged["ots_before"], label="Before", marker='o')
    plt.plot(merged["researchdate"], merged["ots_after"], label="After", marker='s')
    plt.title(f"OTS по дням для бренда {brand_id} (категория {cat_delivery})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"brand_{brand_id}_trend.png"))
    plt.close()

def print_queries_for_anomaly(original_df, subject_id, research_date, anomaly_pairs):
    if anomaly_pairs.empty:
        print("Нет аномальных пар.")
        return
    if (subject_id, research_date) not in set(zip(anomaly_pairs["SubjectID"], anomaly_pairs["researchdate"])):
        print(f"Респондент {subject_id} за {research_date} не отмечен как аномальный.")
        return
    queries = original_df[(original_df["SubjectID"] == subject_id) & 
                          (original_df["researchdate"] == research_date)]["QueryText"].unique()
    print(f"Поисковые запросы респондента {subject_id} за {research_date}:")
    for q in queries:
        print(f"  - {q}")

# ПОМЕСЯЧНЫЕ ГРАФИКИ 
def build_monthly_plots(before_df, after_df, anomaly_pairs):
    months = sorted(before_df["month"].unique())
    for month in months:
        before_m = before_df[before_df["month"] == month]
        after_m = after_df[after_df["month"] == month]
        pairs_m = anomaly_pairs.copy()
        if not pairs_m.empty:
            pairs_m["researchdate"] = pd.to_datetime(pairs_m["researchdate"])
            pairs_m["month"] = pairs_m["researchdate"].dt.to_period("M").astype(str)
            pairs_m = pairs_m[pairs_m["month"] == month]
        else:
            pairs_m = pd.DataFrame()

        # total OTS
        before_day = before_m.groupby("researchdate")["ots"].sum().reset_index(name="before")
        after_day = after_m.groupby("researchdate")["ots"].sum().reset_index(name="after")
        merged = before_day.merge(after_day, on="researchdate", how="outer").fillna(0)
        plt.figure(figsize=(16,6))
        plt.plot(merged["researchdate"], merged["before"], label="Before")
        plt.plot(merged["researchdate"], merged["after"], label="After")
        plt.legend(); plt.xticks(rotation=45)
        plt.title(f"Total OTS - {month}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{month}_total_ots.png"))
        plt.close()

        # category change
        before_cat = before_m.groupby("CategoryNameDelivery")["ots"].sum().reset_index(name="before")
        after_cat = after_m.groupby("CategoryNameDelivery")["ots"].sum().reset_index(name="after")
        merged_cat = before_cat.merge(after_cat, on="CategoryNameDelivery", how="outer").fillna(0)
        merged_cat["change_pct"] = 100 * (merged_cat["after"] - merged_cat["before"]) / (merged_cat["before"]+1e-9)
        merged_cat = merged_cat.sort_values("change_pct")
        plt.figure(figsize=(16,8))
        plt.barh(merged_cat["CategoryNameDelivery"].astype(str), merged_cat["change_pct"])
        plt.title(f"Category OTS Change (%) - {month}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{month}_category_change.png"))
        plt.close()

        # daily anomaly count
        if not pairs_m.empty:
            counts = pairs_m.groupby("researchdate").size().reset_index(name="count")
            plt.figure(figsize=(16,6))
            plt.bar(counts["researchdate"].astype(str), counts["count"])
            plt.xticks(rotation=45)
            plt.title(f"Daily Anomaly Count - {month}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{month}_anomaly_count.png"))
            plt.close()

# MAIN
def main():
    ensure_dirs()
    print("Загрузка данных")
    df_raw = load_data()
    print(" Подготовка данных")
    filtered = prepare_data(df_raw)
    print("Агрегация daily_ots")
    daily = build_daily_ots(filtered)
    print("Выявление аномалий максимальным разрывом")
    anomalies = detect_anomalies_gap(daily)
    print(" Формирование файлов удаления")
    anomaly_pairs = save_anomalies(anomalies)
    save_anomaly_reasons(anomalies)
    print(" Удаление аномальных дней")
    cleaned = remove_anomalies(filtered, anomaly_pairs)
    print("Обязательные графики (п.8.1)")
    plot_total_ots_before_after(filtered, cleaned)
    plot_category_ots_change(filtered, cleaned)
    plot_daily_anomaly_count(anomaly_pairs)

    print("Помесячные графики")
    build_monthly_plots(filtered, cleaned, anomaly_pairs)

    print("Дополнительные аналитические графики (п.8.2)")
    for demo in ["Пол", "Возраст", "Регион", "Федеральный_округ"]:
        if demo in filtered.columns:
            plot_change_by_demographic(filtered, cleaned, demo, PLOTS_DIR)
    for res in ["ResourceName", "ResourceType", "Platform", "UseType"]:
        if res in filtered.columns:
            plot_change_by_resource_feature(filtered, cleaned, res, PLOTS_DIR)

    print(" Пример вывода поисковых запросов")
    if not anomaly_pairs.empty:
        first_anom = anomaly_pairs.iloc[0]
        print_queries_for_anomaly(df_raw, first_anom["SubjectID"], first_anom["researchdate"], anomaly_pairs)
    else:
        print("Нет аномалий.")

    print(" ВСЁ! Результаты в папке output/ ")

if __name__ == "__main__":
    main()
