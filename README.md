# PN2019

### 文件说明

- evaluate_sepsis_score.py 官方给的算utility的文件 使用python3 evaluate_sepsis_score.py labels.zip predictions.zip 执行 zip文件由get_sepsis_test生成
- train_LSTM.py 用来训练LSTM的自动脚本 
- get_sepsis_score.py 用来预测的脚本 里面有特征工程和一些辅助函数 load模型和预测的函数也在里面 需要修改里面的模型名字用于加载不同实验 位置在560行 修正方法也可以在这里改
- get_sepsis_test.py 自制版drive.py 这里可以修改测试文件的量 python3 get_sepsis_test.py 1 50 执行 第一个数字是测试类型 
    - 0 代表全是正常人 
    - 1 代表全是病人 
    - 2 代表比例为7.8%的病人正常人混合样本 
    - 3 需要在数量位置传入新参数trainingA/p019094.psv 代表预测当前特指的样本
    - 11 测试集1含有1000个随机出的样本
    - 22 测试集2含有1000个随机出的样本
    - 55 测试集3含有500个随机出的样本
    - 555 测试集4含有500个随机出的样本
    - 911 test the same cases as official drive.py
    - 
- test_doc.txt 自动记录每次LSTM训练后的模型特征
- data/ 此目录下有trainingA 和trainingB 两个子文件夹 分别存放原数据即可 用于训练的数据是比较大的pkl 可以U盘拷
- src 暂时么得用 路径没改 用于备份
- kw_copy1是最新notebook 原notebook用于并行操作

##### 先执行get_sepsis_test.py 再执行 evaluate_sepsis_score.py 即可得到结果
