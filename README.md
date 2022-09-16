# SpeechEmotionAfterFusion
目前把各种尝试的代码都给删了，只保留了先融合和后融合核心的网络结构，注释还没有全部完善，还需要后续补充。因为debug修改方便的需要，所以代码层次不是很清晰。
## main.py
- 训练模型的入口脚本，可以通过传参，也可以直接改变默认值使用不同配置训练（但是目前有些还不支持），目前默认是使用多模态信息，通过ADD的fusion方式进行先融合和后融合。
## LoadData.py
- 根据batch_size读取训练和测试数据,为了debug方便，通过注释选择使用不同的特征，目前默认是Bert+IS13仿射到100维的特征。
## models.py
- 包含3个主要部分，plot_matrix：绘图函数；Early_fusion：模型类；train_and_test_earlyfusion：训练以及测试函数。
## try_models_advanced.py
- 包含2个主要部分，After_fusion：模型类；train_and_test_afterfusion：训练以及测试函数。
## data_analyze.py
- 用于数学分析，目前仅仅包含PCA的函数。
