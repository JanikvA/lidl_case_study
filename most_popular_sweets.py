from operator import contains
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
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


def main():
    candy_df = pd.read_csv("candy-data.csv")
    candy_df = candy_df.sort_values(by=["winpercent"], ascending=False)
    pd.set_option("display.max_columns", None)
    candy_df.head(5).to_html("most_popular_candybrands.html")
    # features excludes the sugar- and price-percentile columns along the competitor name and winpercent
    features = [
        "chocolate",
        "fruity",
        "caramel",
        "peanutyalmondy",
        "nougat",
        "crispedricewafer",
        "hard",
        "bar",
        "pluribus",
    ]
    feature_correlation(candy_df.drop(["competitorname"], axis=1))
    correlation_plot_winpercent(candy_df.drop(["competitorname"], axis=1))
    print(f"{len(candy_df)=}")
    print(f"{len(candy_df[features].drop_duplicates())=}")
    print(
        f"{len(candy_df.drop(['winpercent', 'competitorname'], axis=1).drop_duplicates())=}"
    )
    num_features = candy_df[features].sum(axis=1)
    candy_df["num_features"] = num_features
    print(f"Avergae number of features: {num_features.mean()} +- {num_features.std()}")

    # combinations
    contains_dict = {
        "features": [],
        "mean_winpct": [],
        "std_winpct": [],
        "n_samples": [],
        "n_feat": [],
        "mean_n_feat": [],
        "std_n_feat": [],
    }
    for num in range(1, 6):
        all_combinations = itertools.combinations(features, num)
        for combi in all_combinations:
            _list_ver = list(combi)
            mean_winpct = (
                candy_df[_list_ver + ["winpercent"]]
                .groupby(_list_ver)
                .mean()["winpercent"]
            )
            std_winpct = (
                candy_df[_list_ver + ["winpercent"]]
                .groupby(_list_ver)
                .std()["winpercent"]
            )
            mean_n_features = (
                candy_df[_list_ver + ["num_features"]]
                .groupby(_list_ver)
                .mean()["num_features"]
            )
            std_n_features = (
                candy_df[_list_ver + ["num_features"]]
                .groupby(_list_ver)
                .std()["num_features"]
            )
            n_samples = candy_df[_list_ver].groupby(_list_ver)[_list_ver[0]].count()
            try:
                ind = (1,) * num
                feat = "+".join(_list_ver)
                mean = mean_winpct.loc[ind]
                std = std_winpct.loc[ind]
                n_samp = n_samples.loc[ind]
                mean_n_feat = mean_n_features.loc[ind]
                std_n_feat = std_n_features.loc[ind]
                contains_dict["features"].append(feat)
                contains_dict["mean_winpct"].append(mean)
                contains_dict["std_winpct"].append(std)
                contains_dict["n_samples"].append(n_samp)
                contains_dict["mean_n_feat"].append(mean_n_feat)
                contains_dict["std_n_feat"].append(std_n_feat)
                contains_dict["n_feat"].append(len(combi))
            except KeyError:
                continue
    contains_df = pd.DataFrame(contains_dict)
    contains_df = contains_df.sort_values(by=["n_samples"], ascending=False)
    contains_df.head(10).to_html("most_common_combinations.html")
    contains_df = contains_df[contains_df["n_samples"] >= 10].sort_values(
        by=["mean_winpct"], ascending=False
    )
    contains_df.head(10).to_html("most_popular_combinations.html")
    contains_df = contains_df[contains_df["n_feat"] == 1].sort_values(
        by=["mean_winpct"], ascending=False
    )
    contains_df.head(10).to_html("winpct_per_feature.html")

    # histplot winpct dist
    ax = sns.histplot(candy_df["winpercent"], color="blue", edgecolor="red")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("winpercent_dist_histplot.png")
    plt.close()

    # histplot num_features
    sns.histplot(num_features)
    plt.savefig("num_feature_histplot.png")
    plt.close()

    # boxplot
    ax = sns.boxplot(
        data=candy_df,
        x="num_features",
        y="winpercent",
        color="blue",
        whiskerprops={"color": "red"},
        capprops={"color": "red"},
        medianprops={"color": "red"},
    )
    sns.swarmplot(data=candy_df, x="num_features", y="winpercent")
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
            color="red",
            bbox=dict(facecolor="yellow", alpha=1),
        )
    plt.savefig("num_feature_winpercent_boxplot.png")
    plt.close()


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


if __name__ == "__main__":
    plt.rcParams.update({"figure.autolayout": True})
    from qbstyles import mpl_style

    mpl_style(dark=True)
    main()
    print("\n### Linear Regression ###")
    lin_reg()
    print("\n### Decision Tree ###")
    decision_tree()
