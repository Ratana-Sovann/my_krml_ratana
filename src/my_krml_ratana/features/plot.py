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



def plot_cities(df, lat, long, city=None, interactive=True):
    """
    Plots city locations on a map using either static or interactive visualization.
    
    Parameters:
    - df (DataFrame): The DataFrame containing city data.
    - lat (str): Column name for latitude values.
    - long (str): Column name for longitude values.
    - city (str or None): Optional column name for city names. Defaults to None.
    - interactive (bool): If True, generates an interactive map using Folium; otherwise, static 	using Matplotlib.
    
    Returns:
    - Folium Map (if interactive=True)
    - Matplotlib plot (if interactive=False)
    """

    # Import packages
    import matplotlib.pyplot as plt
    import folium

    if interactive:
        # Center the map on the average latitude and longitude
        map_center = [df[lat].mean(), df[long].mean()]
        city_map = folium.Map(location=map_center, zoom_start=5)
        
        # Add city markers to the map
        for _, row in df.iterrows():
            popup_text = row[city] if city else f"Lat: {row[lat]}, Long: {row[long]}"
            folium.Marker(
                location=[row[lat], row[long]],
                popup=popup_text
            ).add_to(city_map)
        
        return city_map
    else:
        # Static map with matplotlib
        plt.figure(figsize=(10, 6))
        plt.scatter(df[long], df[lat], color="red", zorder=5)
        
        # Annotate city names if available
        if city:
            for _, row in df.iterrows():
                plt.text(row[long], row[lat], row[city], fontsize=10)
        
        plt.title("City Locations" if city else "Location Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()



def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()



def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

