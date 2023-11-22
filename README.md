# AutoEncoders
R code to train autoencoders with focus on dimensionality reduction

Inspired by [this blog post](https://gradientdescending.com/pca-vs-autoencoders-for-dimensionality-reduction/) I had a look at the comparison of PCA and autoencoders based on the R package [keras](https://cran.r-project.org/web/packages/keras/vignettes/index.html) which offers
an interface to the python package. As result of my own analyses, I added corresponding code using the R package [torch](https://torch.mlverse.org/) as it has no dependency on python which can be handy at times. Finally, I had a look at the R packahe [dimRed](https://cran.r-project.org/web/packages/dimRed/) 
that contains an autoencoder based on keras.
