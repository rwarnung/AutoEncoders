## from https://www.r-bloggers.com/2018/07/pca-vs-autoencoders-for-dimensionality-reduction/
## orig: https://gradientdescending.com/page/3/

library(tidyverse)
library(plotly) ## fro 3D plots
library(tictoc) ## for timing
suppressPackageStartupMessages(library(keras)) ## neural network in keras (based on python)
## installation https://tensorflow.rstudio.com/install/

library(dimRed) ## autoencoder based on keras
library(torch) ## neural network in keras (based on C)

data(ais, package="DAAG")

####### PCA first #############

# standardise
minmax <- function(x) (x - min(x))/(max(x) - min(x))
x_train <- apply(ais[,1:11], 2, minmax)

# PCA
pca <- prcomp(x_train)

# plot cumulative sd explained
ggplot(data.frame(nrPCA = 1:11, SDEV_EXPLAINED = cumsum(pca$sdev)), aes(nrPCA,SDEV_EXPLAINED)) + geom_line()

## plotting first 2 pcs
ggplot(as.data.frame(pca$x), aes(x = PC1, y = PC2, col = ais$sex)) + geom_point()

## plotting first 3 pcs with plotly
pca_plotly <- plot_ly(as.data.frame(pca$x), x = ~PC1, y = ~PC2, z = ~PC3, color = ~ais$sex) |> add_markers()
pca_plotly

########## autoencoder using keras ################

# set training data
x_train <- as.matrix(x_train)
# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 6, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 2, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))  ## the whole data is the output
# view model layers
summary(model)

# compile model
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

nr_epochs = 1000 # 3000

# fit model
model %>% fit(
  x = x_train, 
  y = x_train, 
  epochs = nr_epochs, 
  verbose = 0
)

# evaluate the performance of the model
mse.ae <- evaluate(model, x_train, x_train)
mse.ae

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)

# use summary to see the model structure:
summary(intermediate_layer_model)

## predict on encoder
intermediate_output <- predict(intermediate_layer_model, x_train)

ggplot(data.frame(PC1 = intermediate_output[,1], PC2 = intermediate_output[,2]), aes(x = PC1, y = PC2, col = ais$sex)) + geom_point()

## fit the model again with 3 nodes in the bottleneck layer:

model3 <- keras_model_sequential()
model3 %>%
  layer_dense(units = 6, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 3, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 6, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))

# summar of model
summary(model3)

# compile model
model3 %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

# fit model
model3 %>% fit(
  x = x_train, 
  y = x_train, 
  epochs = nr_epochs,
  verbose = 0
)

# evaluate the model
evaluate(model3, x_train, x_train)

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model3$input, outputs = get_layer(model3, "bottleneck")$output)
intermediate_output <- predict(intermediate_layer_model, x_train)

# plot the reduced dat set
aedf3 <- data.frame(intermediate_output)
colnames(aedf3) = paste0("AE",1:3)
ae_plotly <- plot_ly(aedf3, x = ~AE1, y = ~AE2, z = ~AE3, color = ~ais$sex) %>% add_markers()
ae_plotly

#### autoencoder using the dimRed package ##############
## examples: https://cran.r-project.org/web//packages/dimRed/vignettes/dimensionality-reduction.pdf

## embed the data using an autoencoder based on keras; only gradient decent supported
emb <- embed(x_train, "AutoEncoder", ndim=3, n_hidden = c(6,3,6), activation = rep("tanh",3), learning_rate= 0.001, weight_decay=0, n_steps = nr_epochs)

## measure the quality using the RMSE of the reconstruction, squared for comparability
quality(emb, "reconstruction_rmse")^2

x_train_embed = getDimRedData(emb)

aedf4 <- as.data.frame(x_train_embed)
ae_plotly <- plot_ly(aedf4, x = ~AE1, y = ~AE2, z = ~AE3, color = ~ais$sex) %>% add_markers()
ae_plotly

##### autoencoder using torch
## example: https://anderfernandez.com/en/blog/how-to-create-neural-networks-with-torch-in-r/

# Set seed for reproducibility
set.seed(42)
# Convert data to torch tensors
x_train_tensor <- torch_tensor(x_train)

# Define the network
# initialize defines what happens if an instance of the new class is generated.
# forward defines the method for the neural network
compression_dim = 3
myAutoEncoder1 <- nn_module(
  "MyAutoencoder",
  
  initialize = function() {
    self$encoder <- nn_sequential(
      nn_linear(ncol(x_train), 6),
      nn_tanh(),
      nn_linear(6, compression_dim), ## bottleneck
      nn_tanh()
    )
    
    self$decoder <- nn_sequential(
      nn_linear(compression_dim, 6),  ## bottleneck
      nn_tanh(),
      nn_linear(6, ncol(x_train))
    )
  },
  
  forward = function(x) {
    x <- self$encoder(x)
    x <- self$decoder(x)
    x
  }
)

# Create an instance of the model
autoencoder <- myAutoEncoder1()

# Set up the loss function and optimizer
# the learning rate corresponds to the default in keras for comparability
criterion <- nn_mse_loss()
optimizer <- optim_adam(autoencoder$parameters, lr = 0.001) 

# Training loop

for (epoch in 1:nr_epochs) {
    # Forward pass
  y_pred <- autoencoder(x_train_tensor)
    # Compute loss
  loss <- criterion(y_pred, x_train_tensor)
    # Backward pass and optimization
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
  
  # Print progress
  if (epoch %% 10 == 0) {
      cat(" Epoch:", epoch ,"Loss: ", loss$item(), "\n")
  }
}

# put model in eval model (important if you use e.g. dropout)
autoencoder$eval()

# get encoder:
encoded_train = autoencoder$encoder(x_train_tensor)
encoded_train <- as_array(encoded_train)

encoded_train = data.frame(encoded_train)
colnames(encoded_train) = paste0("AE", 1:compression_dim)

pca_plotly2 <- plot_ly(encoded_train, x = ~AE1, y = ~AE2, z = ~AE3, color = ~ais$sex) |> add_markers()
pca_plotly2

## define function for torch encoder:

torch_autoencoder = function(compression_dim, eval_frequ = 500){
  myAutoEncoder1 <- nn_module(
    "MyAutoencoder",
    
    initialize = function() {
      self$encoder <- nn_sequential(
        nn_linear(ncol(x_train), 6),
        nn_tanh(),
        nn_linear(6, compression_dim), ## bottleneck
        nn_tanh()
      )
      
      self$decoder <- nn_sequential(
        nn_linear(compression_dim, 6),  ## bottleneck
        nn_tanh(),
        nn_linear(6, ncol(x_train))
      )
    },
    
    forward = function(x) {
      x <- self$encoder(x)
      x <- self$decoder(x)
      x
    }
  )
  # Create an instance of the model
  autoencoder <- myAutoEncoder1()
  # Set up the loss function and optimizer
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(autoencoder$parameters, lr = 0.001)
  
  # Training loop
  for (epoch in 1:nr_epochs) {
    # Forward pass
    y_pred <- autoencoder(x_train_tensor)
    # Compute loss
    loss <- criterion(y_pred, x_train_tensor)
    # Backward pass and optimization
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
    
    # Print progress
    if (epoch %% eval_frequ == 0) cat(" Epoch:", epoch ,"Loss: ", loss$item(), "\n")
  }
  
  autoencoder$eval()
  
  recon_train = autoencoder(x_train_tensor)
  recon_train <- as_array(recon_train)
  return(recon_train)
}

########### analysis of reconstruction error ##########
## reconstruction with k = 1:10
## only k=1:5 for autoencoders

# pCA reconstruction
## X_hat = X*rot + mu
pca.recon <- function(pca, x, k){
  mu <- matrix(rep(pca$center, nrow(pca$x)), nrow = nrow(pca$x), byrow = T)
  recon <- pca$x[,1:k] %*% t(pca$rotation[,1:k]) + mu
  mse <- mean((recon - x)^2)
  return(list(x = recon, mse = mse))
}

tic()
PCA.mse <- rep(NA, 10)
for(k in 1:10){
  pca <- prcomp(x_train)
  PCA.mse[k] <- pca.recon(pca, x_train, k)$mse
}
cat("PCA")
toc()


## nr epochs for all autoencoders
nr_epochs = 4000 

tic()
karas.ae.mse <- rep(NA, 5)
for(k in 1:5){
  modelk <- keras_model_sequential()
  modelk %>%
    layer_dense(units = 6, activation = "tanh", input_shape = ncol(x_train)) %>%
    layer_dense(units = k, activation = "tanh", name = "bottleneck") %>%
    layer_dense(units = 6, activation = "tanh") %>%
    layer_dense(units = ncol(x_train))
  
  modelk %>% compile(
    loss = "mean_squared_error", 
    optimizer = "adam"
  )
  
  modelk %>% fit(
    x = x_train, 
    y = x_train, 
    epochs = nr_epochs,
    verbose = 0
  )
  
  karas.ae.mse[k] <- unname(evaluate(modelk, x_train, x_train))
}
cat("Keras")
toc()

tic()
torch.ae.mse <- rep(NA, 5)
for(k in 1:5){
  torch_model_k = torch_autoencoder(k)
  torch.ae.mse[k] <- mean((x_train - torch_model_k)^2)
}
cat("torch")
toc()

df <- data.frame(k = c(1:10, 1:5, 1:5), mse = c(PCA.mse, karas.ae.mse,torch.ae.mse), method = c(rep("pca", 10), rep("keras_autoencoder", 5), rep("torch_autoencoder", 5)))
ggplot(df, aes(x = k, y = mse, col = method)) + geom_line()


