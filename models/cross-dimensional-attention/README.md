# Cross-Dimensional-Attention Model

## [CDSA: Cross-Dimensional Self-Attention for Multivariate, Geo-tagged Time Series Imputation](https://arxiv.org/pdf/1905.09904.pdf)
_Jiawei Ma, Zheng Shou, Alireza Zareian, Hassan Mansour, Anthony Vetro, Shih-Fu Chang_

Many real-world applications involve multivariate, geo-tagged time series data: at
each location, multiple sensors record corresponding measurements. For example,
air quality monitoring system records PM2.5, CO, etc. The resulting time-series
data often has missing values due to device outages or communication errors. In
order to impute the missing values, state-of-the-art methods are built on Recurrent
Neural Networks (RNN), which process each time stamp sequentially, prohibiting
the direct modeling of the relationship between distant time stamps. Recently, the
self-attention mechanism has been proposed for sequence modeling tasks such as
machine translation, significantly outperforming RNN because the relationship between each two time stamps can be modeled explicitly. In this paper, we are the first
to adapt the self-attention mechanism for multivariate, geo-tagged time series data.
In order to jointly capture the self-attention across multiple dimensions, including
time, location and the sensor measurements, while maintain low computational
complexity, we propose a novel approach called Cross-Dimensional Self-Attention
(CDSA) to process each dimension sequentially, yet in an order-independent manner. Our extensive experiments on four real-world datasets, including three standard
benchmarks and our newly collected NYC-traffic dataset, demonstrate that our
approach outperforms the state-of-the-art imputation and forecasting methods. A
detailed systematic analysis confirms the effectiveness of our design choices.

![image](https://drive.google.com/uc?export=view&id=1U-N4c0d3w-pTYc3cFd3iWVbaaFWinNb7)
