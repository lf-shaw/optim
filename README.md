# optim 模块

optim是基于Mosek底层solver开发，用于实现组合优化功能的Python Library。
其中MosekOptimizer作为最底层class提供标准化的优化建模与优化记录功能。
基于这样一个底层optimizer, mean_variance以函数的形式呈现并提供标准化的优化结果输出。

