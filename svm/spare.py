def rfr(train_otu, train_age, test1_otu, test1_age, test2_otu, test2_age):

    print "Running Random Forest with all singletons"

    rf = ensemble.RandomForestRegressor(n_estimators = 1000)

    rf.fit(hsingle_train, hsingle_age)

    rf_importance = rf.feature_importances_

    rev_ranked_imp, rev_ranked_otu = zip(*sorted(zip(rf_importance, otu_heads)))

    rev_ranked_otu = list(rev_ranked_otu)
    rev_ranked_otu.reverse()

    print_lst(rev_ranked_otu)

    hsingle_pred = rf.predict(hsingle_train)

    sp_x, sp_y = spline_fit(hsingle_age,hsingle_pred)

    # print sp_x
    # print sp_y

    fig = plt.figure(0)
    plt.clf()
    plt.plot(hsingle_age, hsingle_pred, '.', color='black')
    plt.plot(sp_x, sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    fig.savefig("Rfr_n1000_maxNone_spline_healthy_single")

    htwin_pred = rf.predict(htwin_train)

    twin_sp_x, twin_sp_y = spline_fit(htwin_age, htwin_pred)

    fig = plt.figure(0)
    plt.clf()
    plt.plot(htwin_age, htwin_pred, '.', color='black')
    plt.plot(twin_sp_x, twin_sp_y)
    plt.xlabel("Biological Age")
    plt.ylabel("Microbiota Age")
    fig.savefig("Rfr_n1000_maxNone_spline_healthy_twins")