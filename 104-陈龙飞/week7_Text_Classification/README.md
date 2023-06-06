项目细节：

1.nn_pipline新增analyse.py和predict.py

	1.1 analyse.py用于数据分析
	
		1)统计正负样本数，输出到控制台
		
		2)将review的字符数统计情况生成为str_lens.csv，保存到data目录下
		
			经查看str_lens.csv，选取40作为文本平均长度
			
		3)生成训练集和测试集
		
			根据csv生成json，训练集和测试集比例为3:1
			
	1.2 predict.py 用于模型训练完成后的预测100条耗时
	
		将耗时写入日志文件【main_模型训练和评估日志.log】
		
		和最终生成的表格【模型效果对比.csv】中
		
2.config.py

	2.1 "epoch": 5
	
		为节省训练时间，将epoch由原来的15改为5
		
	2.2 新增【日志保存目录的参数】和【最终生成的结果文件保存目录的参数】
	
		详细内容可去config.py查看
		
3.main.py

	新增save_model_weights()方法和build_result_contrast_file()方法
	
	3.1 save_model_weights()
	
		用于将训练完成的模型参数按配置的各项参数递归建立目录，保存到对应目录中
		
		比如：
		
			Config["model_type"] = "fast_text"
			
			Config["learning_rate"] = 1e-3
			
			Config["hidden_size"] = 64
			
			Config["batch_size"] = 64
			
			Config["pooling_style"] = "max"
			
		就会依次建立各级目录：output/fast_text/1e-3/hidden_64/batch_64/max
		
			将epoch5.pth保存到该目录下
			
	3.2 build_result_contrast_file()
	
		用于将训练完成的各模型的对比结果总结成表格输出
		
		表格名为【模型效果对比.csv】
		
			保存了各模型的配置及对应的准确率和预测100条耗时的时间（单位：秒）
			
		表格保存在nn_pipline/output目录下
		
	3.3 日志文件【main_模型训练和评估日志.log】
	
		通过logging.basicConfig的参数filename指定生成的日志文件
		
		将原来输出到控制台的日志信息都保存到指定的日志文件【main_模型训练和评估.log】里
		
		日志保存在nn_pipline/log目录下
		
4.超参数的网格搜索

	4.1 bert系列的模型未加入到网格搜索训练
	
		经测试，bert模型训练半轮得44分钟时间，训练1轮近1个半小时
		
		训练5轮得7.5个小时，其他以bert为基础的模型训练时间只会更长
		
		如果把所有和bert相关的模型都加入网格搜索，训练时间至少2个月以上！
		
		为节省训练时间，未将bert系列的模型加入网格训练，但bert的代码也能跑，已测试
		
		也上传了bert模型训练半轮的时间截图，可验证
		
	4.2 将model.py中除bert系列外的各个模型都加入到网格搜索训练
	
		训练时长约10小时，日志文件【main_模型训练和评估日志.log】中记录了各轮训练的时间戳，可验证
	
