# fashion-mnist
## 介绍(Background)
------
通过构建一个分类器，在Fashion-MNIST数据集上测试下。
构建的网络模型包含5个conv层和3个fc层，网络各层参数见下表：
<br>
<table>
<thead><tr><th>名称</th><th>特征图大小</th><th>说明</th></tr></thead>
        <tr>
            <td>输入图像</td>
            <td>28x28x1</td>
            <td>输入图像为灰度图像</td>
        </tr>
         <tr>
            <td>conv1</td>
            <td>28x28x64</td>
            <td>每个conv层包含一个卷积层+relu层+max pooling层。曾使用过batch normalization，但是效果不好(可能是还没收敛，毕竟CPU实在太慢了，所以没使用)</td>
        </tr>
         <tr>
            <td>conv2</td>
            <td>28x28x64</td>
            <td>同上</td>
        </tr>
         <tr>
            <td>conv3</td>
            <td>14x14x64</td>
            <td>同上</td>
        </tr>
         <tr>
            <td>conv4</td>
            <td>14x14x64</td>
            <td>同上</td>
        </tr>
         <tr>
            <td>conv5</td>
            <td>7x7x64</td>
            <td>同上</td>
        </tr>
          <tr>
            <td>fc1</td>
            <td>512</td>
            <td>一个fc层包含一个全连接层+relu层。fc1曾添加过dropout，但此处效果不好，因此最后弃用了。</td>
        </tr>
          <tr>
            <td>fc2</td>
            <td>512</td>
            <td>同上</td>
        </tr>
          <tr>
            <td>fc3</td>
            <td>10</td>
            <td>输出层</td>
        </tr>
</table>
<br>
限于时间关系，没有充分训练，**目前，step=21000（即epoch=21000/(60000/100)=35）,在Fashion-MNIST测试集上的准确率为91.9% 。**

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
