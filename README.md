**Description and motivation for the model**

Compared to a regular neural network where the weights are real numbers, in the bayesian neural network the weights W are treated as random variable with a prior and posterior distribution.
The goal is to model epistemic uncertainty (model uncertainty) thatstems from lack of knowledge or data,
which can be reduced by gathering more information or improving the model.
In more simple terms, if the model has not had enough training data, or the model is not sufficiently deep,
it should be able to say "I dont know", although it will be expressed as a probability distribution with high variance.
When we run inference on the model, we sample the weights to get a distribution for the output. Low variance mean the model is certain.

In this case the output is just up/down (1/0), trying to predict whether the stock market will move up or down at the next time step.
The motivation behind this is to be able to have the model say "I have seen this pattern many times, I am 90% sure the market will go up",
and in other cases "I dont see any recognizable patterns here that I have seen before".

**Results**

For stock market data the amount of aleatoric uncertainty(represents intrinsic randomness or noise in a system that cannot be reduced),
in the data is overwhelming and making the bayesian neural network quite useless. To be able to model epistemic uncertainty the intrinsic noise in the data would need to be reduced.

As we can see from the training results loss on the validation data the loss is barely decreasing after each epoch and increasing the neurons in the hidden layer from 8 to 56 
doesnt improve the results at all. Could also be KL regularization is too strong, forcing the model toward uncertainty

```Training 1 hidden layer, 8 neurons
Epoch 1/30  train_loss=0.529242  train_bce_sum=1649712.845  kl_sum=205087451.641  val_loss=0.524234
Epoch 2/30  train_loss=0.522983  train_bce_sum=1630194.187  kl_sum=228611062.111  val_loss=0.517523
Epoch 3/30  train_loss=0.516731  train_bce_sum=1610699.663  kl_sum=249966183.343  val_loss=0.511253
Epoch 4/30  train_loss=0.511980  train_bce_sum=1595884.656  kl_sum=265052046.662  val_loss=0.510954
Epoch 5/30  train_loss=0.509713  train_bce_sum=1588814.863  kl_sum=277527110.739  val_loss=0.506460

Training 1 hidden layer, 56 neurons
Epoch 1/5  train_loss=0.529662  train_bce_sum=1650558.126  kl_sum=1651762927.020  val_loss=0.524751
Epoch 2/5  train_loss=0.524881  train_bce_sum=1635667.545  kl_sum=1608410825.938  val_loss=0.520938
Epoch 3/5  train_loss=0.520309  train_bce_sum=1621417.298  kl_sum=1606957994.141  val_loss=0.515928
Epoch 4/5  train_loss=0.516312  train_bce_sum=1608957.166  kl_sum=1606505951.629  val_loss=0.513098
Epoch 5/5  train_loss=0.513034  train_bce_sum=1598743.892  kl_sum=1597226269.820  val_loss=0.509503
```

Features used: ['close_price', 'volume', 'number_of_trades', 'ma_30', 'ma_2', 'vol_30']

**Future improvements**

More input features and better model architecture are also possible improvements to the model that could be done.
Create config file so you dont have to manually edit the python files.
Try decreasing KL regularization

**Tech Stack**


pip install pandas numpy scikit-learn torch

sqlite database used for kline data

python 3.9.0 used


**Project Files**

bnn.py --runs the bayesian neural network with the current model archiecture defined in BNNClassifier class

create_binance_db.py --creates the sqlite databases and tables used for storing Binance data

fetch_alpha_klines.py --appends time series data for all tokens with new data

fetch_alpha_tokens.py --fetch all available tokens to fetch time series data for

**Usage**

Run python bnn.py (have to manually edit in BNNClassifier to change architecture).
It saves the .pt that can then be used

The output of the model will give a mean and standard deviation for every output







