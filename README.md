# fashion-mnist
## 介绍(Background)
------
通过构建一个分类器，在Fashion-MNIST数据集上测试下。
构建的网络模型包含五个卷积层和3个全连接层，网络各层参数见下表：
<br>
<table>
<thead><tr><th>名称</th><th>特征图数量</th><th>核大小</th><th>说明</th></tr></thead>
        <tr>
            <td>输入图像</td>
            <td>28x28x1</td>
            <td>/</td>
            <td>/</td>
        </tr>
</table>
<br>

## 项目依赖(Requirement)
-----

1.Tensorflow;<br>
2.CUDA.(可选)<br>

## 项目文件(File contents)
-----

1.` data` : ` data`文件夹包含文件夹` fashion`，里面放了Fashion-MNIST数据集;<br>
2.`model `: 该文件夹用于保存训练时的模型文件;<br>
3.`trained_model` : 该文件夹用于存放训练完成后的模型;<br>
4.`train.py ` : 运行该py文件来训练模型，训练完成后会进行测试，输出测试错误率.<br>

## 实验环境(Experimental Enveriment)
-----
该项目在Windows 7 x64上运行，所用Tensorflow版本为1.4。因为是用CPU跑的，比较慢，估计还没完全收敛。
