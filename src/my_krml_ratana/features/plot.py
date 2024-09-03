def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Plots a boxplot and a histogram for the specified feature.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the feature.
    feature : str
        The column name in the dataframe for which to plot the boxplot and histogram.
    figsize : tuple, optional
        Size of the figure (default is (12, 7)).
    kde : bool, optional
        Whether to show the density curve on the histogram (default is False).
    bins : int or sequence, optional
        Number of bins for the histogram (default is None).

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create subplots: boxplot on top and histogram on bottom
    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2, 
        sharex=True, 
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize
    )
    
    # Plot the boxplot
    sns.boxplot(
        data=data, 
        x=feature, 
        ax=ax_box, 
        showmeans=True, 
        color="violet"
    )
    
    # Plot the histogram with optional KDE and bins
    sns.histplot(
        data=data, 
        x=feature, 
        kde=kde, 
        ax=ax_hist, 
        bins=bins, 
        palette="winter"
    )
    
    # Add vertical lines for mean and median
    ax_hist.axvline(
        data[feature].mean(), 
        color="green", 
        linestyle="--", 
        label="Mean"
    )
    ax_hist.axvline(
        data[feature].median(), 
        color="black", 
        linestyle="-", 
        label="Median"
    )
    
    # Add legend for the histogram
    ax_hist.legend()
    
    plt.show()


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Creates a barplot with optional percentage labels at the top of the bars.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the feature.
    feature : str
        The column name in the dataframe to plot.
    perc : bool, optional
        Whether to display percentages instead of counts (default is False).
    n : int, optional
        Number of top categories to display. If None, all categories are displayed (default is None).

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    total = len(data[feature])  # Total number of observations
    unique_count = data[feature].nunique()  # Number of unique categories
    
    if n is None:
        figsize = (unique_count + 1, 5)
    else:
        figsize = (n + 1, 5)

    plt.figure(figsize=figsize)
    plt.xticks(rotation=90, fontsize=15)
    
    # Create the barplot
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values()
    )
    
    # Annotate bars with labels
    for p in ax.patches:
        if perc:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )
        else:
            label = int(p.get_height())
        
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        
        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points"
        )
    
    plt.show()


