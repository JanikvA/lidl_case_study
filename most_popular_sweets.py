import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from matplotlib.ticker import MaxNLocator
import itertools


def make_barplot(feat_cdf):
    ax = sns.barplot(feat_cdf, x="combination", y="mean_winpercent", color="blue")
    ax.tick_params(axis="x", labelrotation=25)
    # ax.bar_label(ax.containers[0])
    labels = ax.get_xticklabels()
    for label in labels:
        ax.text(
            label._x,
            30,
            feat_cdf[feat_cdf["combination"] == label.get_text()]["n_samples"].iloc[0],
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="blue",
            bbox=dict(edgecolor="red", facecolor="yellow", alpha=1),
        )
    ax.set_xlabel(" ")
    ax.set_xticklabels(
        [
            "Knusprig",
            "Erdnuss/Mandel",
            "Riegel",
            "Schokolade",
            "Nugat",
            "Karamell",
            "Mehrteilig",
            "Fruchtig",
            "Hart",
        ]
    )
    ax.set_ylabel("Siegerquote [%]")
    plt.savefig("barplot_per_feature.png")
    plt.close()


def feature_correlation(plot_data):
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        plot_data.corr(method="pearson"),
        vmin=-1,
        vmax=1,
        annot=True,
        ax=ax,
        fmt=".2f",
        cmap="plasma",
    )
    ax.set_title(f"Pearson Correleation Matrix", fontsize=20)
    fig.savefig("pearson_correlation_matrix.png")
    plt.close(fig)


def correlation_plot_winpercent(df):
    corr = df.corr(method="pearson")
    temp = corr["winpercent"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.barplot(x=temp[1:], y=temp.index[1:], orient="h", ax=ax, palette="plasma")
    ax.bar_label(ax.containers[0], fmt="%.2f")
    ax.set_title("pearson correleation")
    fig.savefig("pearson_correlation_winpercent.png")
    plt.close(fig)


def make_boxplot(x, y, out_name, x_label=None, x_tick_labels=None):
    ax = sns.boxplot(
        x=x,
        y=y,
        color="blue",
        whiskerprops={"color": "red"},
        capprops={"color": "red"},
        medianprops={"color": "red"},
    )
    sns.swarmplot(x=x, y=y)
    lines = ax.get_lines()
    categories = ax.get_xticks()
    for cat in categories:
        y = round(lines[4 + cat * 6].get_ydata()[0], 1)
        ax.text(
            cat,
            y,
            f"{y}",
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="blue",
            bbox=dict(edgecolor="red", facecolor="yellow", alpha=1),
        )
    if x_label:
        ax.set_xlabel(x_label)
    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel("Siegerquote [%]")
    plt.savefig(out_name)
    plt.close()


def create_contains_df(data, characteristics):
    mean_std_vars = ["winpercent", "num_features", "num_flavours"]
    contains_dict = {
        "combination": [],
        "n_samples": [],
        "n_feat": [],
    }
    for mean_std in ["mean", "std"]:
        contains_dict.update({f"{mean_std}_{var}": [] for var in mean_std_vars})
    for num in range(1, 6):
        all_combinations = itertools.combinations(characteristics, num)
        for combi in all_combinations:
            _list_ver = list(combi)
            ind = (1,) * num
            feat = "+".join(_list_ver)
            n_samples = data[_list_ver].groupby(_list_ver)[_list_ver[0]].count()
            if ind in n_samples or len(ind) == 1:
                n_samp = n_samples.loc[ind]
                contains_dict["combination"].append(feat)
                contains_dict["n_samples"].append(n_samp)
                contains_dict["n_feat"].append(len(combi))
                for val in mean_std_vars:
                    grouped_view = data[_list_ver + [val]].groupby(_list_ver)[val]
                    contains_dict[f"mean_{val}"].append(grouped_view.mean().loc[ind])
                    contains_dict[f"std_{val}"].append(grouped_view.std().loc[ind])
            else:
                continue
    contains_df = pd.DataFrame(contains_dict)
    return contains_df


def main():
    candy_df = pd.read_csv("candy-data.csv")
    # flavours/features exclude the sugar- and price-percentile columns as well as the competitor name and winpercent
    flavours = [
        "chocolate",
        "fruity",
        "caramel",
        "peanutyalmondy",
        "nougat",
    ]
    features = [
        "crispedricewafer",
        "hard",
        "bar",
        "pluribus",
    ]
    print(f"{len(candy_df)=}")
    print(f"{len(candy_df[features].drop_duplicates())=}")
    print(
        f"{len(candy_df.drop(['winpercent', 'competitorname'], axis=1).drop_duplicates())=}"
    )
    num_features = candy_df[features].sum(axis=1)
    candy_df["num_features"] = num_features
    print(f"Avergae number of features: {num_features.mean()} +- {num_features.std()}")
    num_flavours = candy_df[flavours].sum(axis=1)
    candy_df["num_flavours"] = num_flavours
    print(f"Avergae number of flavours: {num_flavours.mean()} +- {num_flavours.std()}")

    # popular combinations
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 1)
    contains_df = create_contains_df(candy_df, features + flavours)
    common_cdf = contains_df.sort_values(by=["n_samples"], ascending=False)
    common_cdf.head(5).to_html("most_common_combinations.html", index=False)
    pop_cdf = contains_df[contains_df["n_samples"] >= 10].sort_values(
        by=["mean_winpercent"], ascending=False
    )
    pop_cdf.head(5).to_html("most_popular_combinations.html", index=False)
    least_cdf = contains_df[contains_df["n_samples"] >= 10].sort_values(
        by=["mean_winpercent"], ascending=True
    )
    least_cdf.head(5).to_html("least_popular_combinations.html", index=False)
    feat_cdf = contains_df[contains_df["n_feat"] == 1].sort_values(
        by=["mean_winpercent"], ascending=False
    )
    feat_cdf.head(10).to_html("winpct_per_feature.html", index=False)
    make_barplot(feat_cdf)

    brands_cdf = candy_df.sort_values(by=["winpercent"], ascending=False)
    brands_cdf.head(5).to_html("most_popular_candybrands.html", index=False)

    # correlation plots
    feature_correlation(candy_df.drop(["competitorname"], axis=1))
    correlation_plot_winpercent(candy_df.drop(["competitorname"], axis=1))

    # co-occurence
    binary_feat_candy_df = candy_df[flavours + features]
    coocc_mat = binary_feat_candy_df.T.dot(binary_feat_candy_df)
    coocc_mat = coocc_mat.apply(lambda row: row / row[row.name], axis=1)
    coocc_mat.to_html("cooccurence_matrix.html", float_format=lambda flt: f"{flt:.2f}")

    # histplot winpct dist
    ax = sns.histplot(candy_df["winpercent"], color="blue", edgecolor="red")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("winpercent_dist_histplot.png")
    plt.close()

    # pairplot for continious variables
    sns.pairplot(
        candy_df,
        vars=["winpercent", "pricepercent", "sugarpercent"],
        hue="chocolate",
        corner=True,
    )
    plt.savefig("pairplot_choco_hue.png")
    plt.close()

    sns.pairplot(
        candy_df, vars=["winpercent", "pricepercent", "sugarpercent"], corner=True
    )
    plt.savefig("pairplot.png")
    plt.close()

    # histplot num_features
    sns.histplot(num_features)
    plt.savefig("num_feature_histplot.png")
    plt.close()

    # histplot num_flavours
    sns.histplot(num_flavours)
    plt.savefig("num_flavours_histplot.png")
    plt.close()

    # boxplot num features/flavours
    make_boxplot(
        candy_df["num_features"],
        candy_df["winpercent"],
        "num_feature_winpercent_boxplot.png",
        "Anzahl der Charakteristika",
    )
    make_boxplot(
        candy_df["num_flavours"],
        candy_df["winpercent"],
        "num_flavours_winpercent_boxplot.png",
        "Anzahl der Geschm??cker",
    )
    # boxplot fruity candy
    make_boxplot(
        candy_df["fruity"],
        candy_df["winpercent"],
        "fruity_winpercent_boxplot.png",
        x_tick_labels=["nicht fruchtig", "fruchtig"],
        x_label=" ",
    )
    make_boxplot(
        (
            (candy_df["fruity"] == 1)
            & (candy_df["hard"] == 0)
            & (candy_df["pluribus"] == 1)
        ),
        candy_df["winpercent"],
        "soft_pluribus_fruity_winpercent_boxplot.png",
        x_tick_labels=["nicht fruchtgummi??hnlich", "fruchtgummi??hnlich"],
    )
    # boxplot cookie like candy
    make_boxplot(
        (candy_df["crispedricewafer"] == 1),
        candy_df["winpercent"],
        "cookie_like_winpercent_boxplot.png",
        x_tick_labels=["nicht keks??hnlich", "keks??hnlich"],
        x_label=" ",
    )
    make_boxplot(
        ((candy_df["crispedricewafer"] == 1) & (candy_df["chocolate"] == 1)),
        candy_df["winpercent"],
        "choco_cookie_like_winpercent_boxplot.png",
        x_tick_labels=["nicht schokokeks??hnlich", "schokokeks??hnlich"],
    )


def lin_reg():
    candy_df = pd.read_csv("candy-data.csv")
    linreg = LinearRegression()
    x_data = candy_df.drop(["winpercent", "competitorname"], axis=1)
    y_data = candy_df["winpercent"]
    linreg.fit(x_data, y_data)
    coef_tups = sorted(zip(x_data.columns, linreg.coef_), key=lambda ele: ele[1])
    print(
        "Coefficients: \n", "\n".join([f"{tup[0]}:{tup[1]:.2f}" for tup in coef_tups])
    )
    y_pred = linreg.predict(x_data)
    metrics = [
        mean_squared_error,
        r2_score,
        explained_variance_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        median_absolute_error,
    ]
    for met in metrics:
        print(f"{met.__name__}: {met(y_data, y_pred)}")
    for var in x_data.columns:
        if var == "sugarpercent" or var == "pricepercent":
            continue
        tmp_x_data = x_data[x_data[var] == 1]
        tmp_y_data = y_data[tmp_x_data.index]
        # print(var,len(tmp_x_data),len(tmp_y_data))
        tmp_r2_score = linreg.score(tmp_x_data, tmp_y_data)
        tmp_pred = y_pred[tmp_x_data.index]
        tmp_mse = mean_squared_error(tmp_y_data, tmp_pred)
        print(
            f"samples with {var}: n_samps = {len(tmp_y_data)} | R2 = {tmp_r2_score} | MSE = {tmp_mse}"
        )
    cv = cross_validate(
        linreg,
        x_data,
        y_data,
        scoring="neg_mean_squared_error",
        cv=RepeatedKFold(n_splits=4, n_repeats=100),
        return_train_score=True,
    )
    print(
        f"4-foldx100 cross validation RMSE test-score (train-score): {cv['test_score'].mean():.2f} +- {cv['test_score'].std():.2f} ({cv['train_score'].mean():.2f} +- {cv['train_score'].std():.2f})"
    )


def decision_tree():
    candy_df = pd.read_csv("candy-data.csv")
    model = DecisionTreeRegressor(max_depth=3)
    x_data = candy_df.drop(["winpercent", "competitorname"], axis=1)
    y_data = candy_df["winpercent"]
    model.fit(x_data, y_data)
    y_pred = model.predict(x_data)
    metrics = [
        mean_squared_error,
        r2_score,
        explained_variance_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        median_absolute_error,
    ]
    for met in metrics:
        print(f"{met.__name__}: {met(y_data, y_pred)}")
    annot_artists = plot_tree(
        model, feature_names=x_data.columns, rounded=True, filled=True
    )
    for a in annot_artists:
        a.set_color("black")
    fig = plt.gcf()
    fig.set_size_inches(5.5, 1.5)
    plt.savefig("tree_plot.png", dpi=300)
    cv = cross_validate(
        model,
        x_data,
        y_data,
        scoring="neg_mean_squared_error",
        cv=RepeatedKFold(n_splits=4, n_repeats=100),
        return_train_score=True,
    )
    print(
        f"4-foldx100 cross validation RMSE test-score (train-score): {cv['test_score'].mean():.2f} +- {cv['test_score'].std():.2f} ({cv['train_score'].mean():.2f} +- {cv['train_score'].std():.2f})"
    )


if __name__ == "__main__":
    plt.rcParams.update({"figure.autolayout": True})
    from qbstyles import mpl_style

    mpl_style(dark=True)

    main()
    print("\n### Linear Regression ###")
    lin_reg()
    print("\n### Decision Tree ###")
    decision_tree()
