## beta-CROWN

Arxiv链接：<https://arxiv.org/abs/2103.06624>

Github仓库：<https://github.com/KaidiXu/Beta-CROWN>

[CROWN](https://arxiv.org/abs/1811.00866)是神经网络验证方向比较有代表性的工作，大概是2018年发表的文章，后来在2020年有了[alpha-CROWN](https://arxiv.org/abs/2011.13824)，随后就是beta-CROWN. 这两部分工作分别提出了不完全的验证算法和完全的验证算法，并且都是可以GPU计算的，两个算法集成起来就是alpha-beta-CROWN，这个工具在[VNN-COMP 2021](https://sites.google.com/view/vnn2021)竞赛上获得了最高分，所以CROWN的这两个后续工作应该算是目前神经网络验证方面的SOTA了. 竞赛中分数较高的还有[VeriNet](https://github.com/vas-group-imperial/VeriNet), [oval](https://github.com/oval-group/oval-bab), [ERAN](https://github.com/eth-sri/eran)，其中只有VeriNet是CPU计算，其他都是GPU计算. 

### 背景

神经网络验证的不完全方法中有两类，一类是通过线性规划LP求输出变量界的，这类方法的主要想法是通过对激活层神经元的上下界做粗略估计，从而用线性约束去松弛原有的非线性约束. 除了ReLU之外，也有工作处理类sigmoid函数的估计，使用各种不同的线性约束等. 但是由于LP求解中使用的变量数多，且复杂度高，因此这类方法的效率比较低.

另一类不完全方法是bound propagation，CROWN就是一种典型bound propagation方法，类似的还有DeepPoly，这些方法都可以用抽象解释的框架来描述. 这类方法是逐层估计神经元的上下界直至输出层，因而在对激活神经元进行估计时，不能使用三角形约束（对于ReLU层），而只能使用一个下界估计（alpha-CROWN就是对下界估计中的斜率alpha也进行优化），但这样带来的最大好处是算法复杂度相对较低，效率比较高，对于n层网络来说，时间复杂度与n的平方成正比.

分支限界法(BaB)是一种有代表性的完全验证方法，但是由于每一次分支需要对神经元状态进行指派，所以会引入很多神经元状态的约束，这些约束很难在bound propagation的过程中得到完整的保持，这样就只能用线性LP来求解，而LP一般认为是和输入规模n的3.5次方成正比，再加上BaB本身的开销，所以完全验证方法的效率比较低. 

beta-CROWN这篇文章提出一种新的bound propagation的验证方法，这种方法可以在传播过程中完整的保持神经元状态的约束，从而可以通过BaB完全化. 同时该方法也可以通过GPU并行加速，实验表明与基于LP的BaB完全验证方法相比，beta-CROWN BaB至多快三个数量级，验证精度也更高. 具体来说，beta-CROWN在传播bound的同时传播拉格朗日乘子，并引入了一个参数beta，只要参数的设置是合理的，最终产生的bound就一定是可靠的. 通过梯度的方法对beta进行优化，就可以使得最终的bound更紧，同时帮助BaB做剪枝.

### 拉格朗日乘子表达神经元分裂约束

通过把BaB过程中引入的神经元状态约束加入到优化问题中，神经网络验证问题可以表达为以下形式

![eq1](https://luan-xiaokun.github.io/assets/images/beta-CROWN/beta-CROWN-eq1.png)

利用类似CROWN的方法，将最后一层激活层用线性约束近似，验证问题转化为

![eq2](https://luan-xiaokun.github.io/assets/images/beta-CROWN/beta-CROWN-eq2.png)

通过引入拉格朗日乘子beta，以及定义矩阵S（S表示哪些神经元状态不确定/激活/未激活），得到以下不等式

![eq3](https://luan-xiaokun.github.io/assets/images/beta-CROWN/beta-CROWN-eq3.png)

第一个不等式是根据约束Z^{(L-1)}的拉格朗日乘子得到的，第二个不等式弱对偶性. 再将z^{(L-1)}展开，可以得到如下形式

![eq4](https://luan-xiaokun.github.io/assets/images/beta-CROWN/beta-CROWN-eq4.png)

该式与第一个公式类似，记\hat{z}前面的矩阵为A^{i}，并且反复使用拉格朗日乘子，就可以将神经网络验证问题转化为如下形式

![thm1](https://luan-xiaokun.github.io/assets/images/beta-CROWN/beta-CROWN-thm1.png)

文章中将这一过程用定理加以证明，风格与当年的CROWN十分类似

由于只要beta大于0，得到的验证结果就是可靠的，例如CROWN就是beta=0时的特例，因此不必对beta做最优化. 文章中使用梯度上升针对beta进行优化（关于beta是concave的）

### 自由变量联合优化

文章中作者还证明了若对beta也进行最优化，以上问题的解与BaB&LP求解问题的最优解相同. 在BaB&LP中，激活层的上下界的估计往往是取定的，例如通过效率较高的不完全验证器得到一个较为粗略的界，再在此基础上进行优化. 上一小节中也是这样子做的，但是实际上beta-CROWN可以针对这些自由变量进行联合优化，这将会产生比LP更紧致的界.

具体来说，为了得到第i层神经元的下界，可以把前i层视为一个新的网络，针对这个网络有一个对应的beta-CROWN最优化问题，其解就是第i层神经元下界的估计，求解这个问题过程中，将会引入新的beta'变量（但是这些变量与要求解的原问题的变量beta无关）. 而对于上界，只需要对第i层输出取反再使用相同的方法即可. 这一点与其他的bound propagation方法十分相似. 

作者对自由变量联合优化的变量数做了估计，对于L层、每层d个神经元的网络来说，变量数是O(L^2d^2)的. 这样的问题规模过于庞大，因此作者采用了类似CNN的变量共享，即同一层的神经元共享beta参数，这样不会对验证器的可靠性产生影响，但能降低求解规模.

### 分支限界法

由于beta-CROWN对神经元分裂约束，也就是神经元状态约束的良好支持，可以将beta-CROWN作为BaB的不完全验证器，而不必使用开销较高的LP求解器. 这样得到的beta-CROWN BaB是完全的，因为当所有的神经元的激活状态都被确定时，beta-CROWN可以得到一个最小值，这一点是CROWN做不到的. 也就是说，beta-CROWN可以检测BaB中某个分支的不可行性，从对偶问题来看，当对偶问题无解时，原问题的解是无界的. 

除了将beta-CROWN完全化，作者还将beta-CROWN BaB不完全化，也就是设定时间限制，将beta-CROWN BaB作为一个不完全求解器.

至于BaB时的分支策略，文章中使用的是做BaB完全验证比较流行的BaBSR和效率更高一些的FSB.

### 实验结果

从实验结果来看，beta-CROWN和beta-CROWN BaB远超了同时期其他验证方法. 

beta-CROWN BaB和10个其他完全求解器在VNN COMP 2020数据集上进行比较，验证时间显著优于其他方法，在10秒内就能验证90%以上的性质

beta-CROWN和5个其他不完全求解器的比较结果也是beta-CROWN显著优于其他方法，但是不知道为什么要特别强调超过了SDP的方法，SDP明明不如PRIMA

此外作者还顺带提了一下VNN COMP 2021，但却没有明说他们获得了第一名，大概是投稿时还没公布结果. 在文章结尾作者还阐述了这篇工作社会影响，表示虽然这样的工作可能会被用来做攻击，但是作者认为还是积极影响更多一些，这也有一点让人摸不到头脑，为什么要在Conclusion加这样一段话，可能是对评审人的疑问的回答吧.

### 总结

这篇文章提出了beta-CROWN，通过拉格朗日乘子来表达神经元的分裂约束，同时通过自由变量联合优化，得到了一个更精确的不完全验证器. 由于beta-CROWN能够表达神经元的分裂约束，又是基于bound propagation的高效验证器，因此可以通过BaB完全化，得到一个高效的完全验证器. 实验结果表明beta-CROWN显著优于其他求解方法.

作者也总结了这篇工作的局限之处，一是只处理了ReLU激活层，对于非分段线性的激活函数，通过BaB保证完全性很困难；二是只考虑Lp范数的输入约束，实际的攻击往往是复杂的非凸的扰动；三是实验部分使用的网络仍然规模有限，无法达到类似ImageNet的规模，不过这也是做神经网络验证的一个普遍问题.

从我个人来看，还有以下几个方面可以继续探索，一是找到更契合的分支策略，一般来说很难有较为普适的策略才对，所以应该可以找到对于某些算法更高效的分支策略；二是能否把拉格朗日乘子的方法用在类sigmoid激活函数上，就像CROWN一样；三是探究自由变量联合优化对求解精度的影响，是否可能替换成开销更低的办法（当然作者应该已经尝试过求解完整问题，发现问题规模太大效率不高）. 另外，看到这样一篇吊打其他方法的sota，感觉做神经网络验证确实越来越难了
