# dsa 课程总结与笔试复习

## 课程总结

开始写这段总结是6月19日，昨天考完了数算的机考，说实话笔试卷子的难度比我预想的要低一些，虽然我不见得会考得有多好（）考场里好多人提前交卷走了，他们应该会比我考得更好一些（）

相比于计概，数算的题目模板性要更强一些，虽然计概也有贪心，dp这种对一类算法的概括，但是它们往往没有标准的写法，而数算不同，无论是笔试还是机考，分析某个问题常常会有一个固定的方法，相应地确定了一种实现的方法（虽然笔试的时候那个prim的写法没怎么见过）。

如果让我用一个词概括数算的学习方法，我会用“积累”。计概中做题更多的是为了提高自己分析问题的能力，如何找到合适的贪心策略，如何写出正确的递推关系式。而数算中更多的是积累和强化一个模板，并在上面添加限制条件或者进行优化。学数算就像是盖一栋房子，一层一层往上垒。所以多码代码，多去理别人代码的逻辑对能力的提升很明显。考计概的时候我的cheat sheet是一点没用，完全不知道该看什么地方。但是考数算的时候cheat sheet确实帮到了我，顺着上边的标准思路AC了一道题。

计概和数算都是硬课，对于一个没有竞赛基础的小白来说，想要学好这两门课需要付出的时间和精力一点不比各个院系的专业课少，但我同时觉得数算不是一门特别需要天赋的课程，和dalao们的差距是可以通过努力弥补的。这是一门付出和收获成强正相关的课程，至少我觉得我的成绩和我的付出是匹配的（）

下面是这门课笔试前的复习资料，基本上整理自gw老师的课件和闫老师的github

## 小寄巧

#### 中缀转后缀

给每一次运算括上一个括号，将运算符挪到**它对应**的括号后面即得



## 数据结构

数据结构(data structure)就是数据的组织和存储形式。描述一个数据结构，需要指出其**逻辑结构**、**存储结构**和**可进行的操作**。

将数据的单位称作“元素”或“结点”。数据结构描述的就是**结点之间的关系**。

#### 逻辑结构

从逻辑上描述结点之间的关系，和数据的**存储方式无关**。

**集合结构**：结点之间没有什么关系，只是属于同一集合。如set。

**线性结构**：除了最靠前的结点，每个结点有唯一前驱结点；除了最靠后的结点，每个结点有唯一后继结点。如list。

**树结构**：有且仅有一个结点称为”根结点”，其没有前驱(父结点)；有若干个结点称为 “叶结点”，没有后继(子结点)；其它结点有唯一前驱，有1个或多个后继。如家谱

**图结构**：每个结点都可以有任意多个前驱和后继，两个结点还可以互为前驱后继。如铁路网，车站是结点。

（都在用**前驱**和**后继**描述，重点是它们的个数和特殊情况）

### 存储结构

数据在物理存储器上存储的方式，大部分情况下指的是数据在内存中存储的方式。

**顺序结构**：结点在内存中**连续存放**，所有结点占据一片**连续的内存空间**。如list。

**链接结构**：结点在内存中**可不连续存放**，每个结点中存有**指针**指向其前驱结点和/或后继结点。如链表，树。

**索引结构**：将结点的**关键字信息**（比如学生的学号）拿出来**单独存储**，并且为每个关键字x配一个指针指向关键字为x的结点，这样便于**按照关键字查找**到相应的结点。

**散列结构**：设置**散列函数**，散列函数以结点的关键字为参数，算出一个结点的存储位置。



**数据的逻辑结构和存储结构无关**

一种逻辑结构的数据，可以用不同的存储结构来存储。

树结构、图结构可以用链接结构存储，也可以用顺序结构存储

线性结构可以用顺序结构存储，也可以用链接结构存储。





## 二分算法

```python
def binarySearch(a,p,key = lambda x : x): 
    L,R = 0,len(a)-1  #查找区间的左右端点，区间含右端点 
    while L <= R: #如果查找区间不为空就继续查找 
        mid = L+(R-L)//2 #取查找区间正中元素的下标 
        if key(p) < key(a[mid]): 
            R = mid - 1  #设置新的查找区间的右端点 
        elif key(a[mid]) < key(p): 
            L = mid + 1  # 设置新的查找区间的左端点 
        else:
            return mid 
    return None 
```

二分法找最优答案的核心思想就是：

如果一个假设的答案**成立**，那就跳着试一个**更优的假设**答案看行不行；

如果一个假设的答案**不成立**,那就跳着试一个**更差的假设**答案看行不行。

必须每次验证假设答案，都可以把假设答案所在的区间缩小为上次的一半。



前提：**单调性**。一个假设答案不成立，则比它更优的假设答案肯定都不成立。



## 线性表

线性表是一个**元素构成的序列**

该序列有**唯一**的**头元素**和**尾元素**，除了头元素外，每个元素都有唯一的前驱元素，除了尾元素外，每个元素都有唯一的后继元素

线性表中的元素**属于相同的数据类型**，即每个元素所占的空间必须相同。

分为**顺序表**和**链表**两种

### 顺序表

即Python的**列表**，以及其它语言中的**数组**

元素在内存中**连续存放**

每个元素都有**唯一序号(下标）**，且根据序号**访问**（包括**读取**和**修改**）元素的时间复杂度是**O(1)**的 --- 随机访问

下标为i的元素前驱下标为i-1，后继下标为i+1

find(),insert(),remove()是O(n)的，其余操作是O(1)的

#### 顺序表的append的O(1)复杂度的实现

总是分配**多于**实际元素个数的空间(容量大于元素个数）

元素个数**小于容量**时，append操作复杂度**O(1)**

元素个数**等于容量**时，append导致**重新分配空间**，且要**拷贝**原有元素到新空间，复杂度**O(n)**

### 链表

元素在内存中**并非连续存放**，元素之间通过指针链接起来

每个结点除了元素，还有next**指针**，指向后继

**不支持随机访问**。访问第i个元素，复杂度为**O(n)**

**已经找到**插入或删除位置的情况下，插入和删除元素的**复杂度O(1)**,且**不需要复制或移动结点**

**有多种形式：**  

l 单链表

l 循环单链表

l 双向链表

l 循环双向链表

#### 单链表

```python
class LinkList:
	class Node: #表结点
		def __init__(self, data, next=None):
			self.data, self.next = data, next
	def __init__(self):
		self.head = self.tail = None
		self.size = 0
	def printList(self): #打印全部结点
		ptr = self.head
		while ptr is not None: 
			print(ptr.data, end=",")
			ptr = ptr.next
	def insert(self,p,data): #在结点p后面插入元素
		nd = LinkList.Node(data,None)
		if self.tail is p:  # 新增的结点是新表尾
			self.tail = nd
		nd.next = p.next
		p.next = nd
		self.size += 1
	def popFront(self): #删除前端元素
		if self.head is None:
			raise \ 
		  Exception("Popping front for Empty link list.")
		else:
			self.head = self.head.next
			self.size -= 1
			if self.size == 0:
				self.head = self.tail = None
	def pushBack(self,data): #在尾部添加元素
		if self.size == 0:
			self.pushFront(data)
		else:
			self.insert(self.tail,data)
	def pushFront(self,data): #在链表前端插入一个元素data
		nd = LinkList.Node(data, self.head)
		self.head = nd
		self.size += 1
		if self.tail is None:
			self.tail = nd
	def clear(self):
		self.head = self.tail = None
		self.size = 0
	def __iter__(self):
		self.ptr = self.head
		return self
	def __next__(self):
		if self.ptr is None:
			raise StopIteration()  # 引发异常
		else:
			data = self.ptr.data
			self.ptr = self.ptr.next
			return data
```

#### 双向链表(双链表）

```python
class DoubleLinkList:
	class _Node:
		def __init__(self, data, prev=None, next=None):
			self.data, self.prev, self.next = data, prev, next
	class _Iterator: 
		def __init__(self,p):
			self.ptr = p
		def getData(self):
			return self.ptr.data
		def setData(self,data):
			self.ptr.data = data
		def __next__(self):
			self.ptr = self.ptr.next
			if self.ptr is None:
				return None
			else:
				return DoubleLinkList._Iterator(self.ptr)
		def prev(self):
			self.ptr = self.ptr.prev
			return DoubleLinkList._Iterator(self.ptr)

	def __init__(self):
		self._head = self._tail = \
			DoubleLinkList._Node(None,None,None)
		self._size = 0
	def _insert(self,p,data):########此处往下重要#########
		nd = DoubleLinkList._Node(data,p,p.next)
		if self._tail is p:  # 新增的结点是新表尾
			self._tail = nd
		if p.next:
			p.next.prev = nd
		p.next = nd
		self._size += 1
	def _delete(self,p):  #删除结点p
		if self._size == 0 or p is self._head:
			raise Exception("Illegal deleting.")
		else:
			p.prev.next = p.next
			if p.next: #如果p有后继
				p.next.prev = p.prev
			if self._tail is p:
				self._tail = p.prev
			self._size -= 1
	def clear(self):
		self._tail = self._head
		self._head.next = self._head.prev = None
		self.size = 0
	def begin(self):
		return DoubleLinkList._Iterator(self._head.next)
	def end(self):
		return None
	def insert(self,i,data): #在迭代器i指向的结点后面插入元素
		self._insert(i.ptr,data)
	def delete(self, i):  # 删除迭代器i指向的结点
		self._delete(i.ptr)
	def pushFront(self,data): #在链表前端插入一个元素
		self._insert(self._head,data)
	def popFront(self):
		self._delete(self._head.next)
	def pushBack(self,data):
		self._insert(self._tail,data)
	def popBack(self):
		self._delete(self._tail)
	def __iter__(self):
		self.ptr = self._head.next
		return self
	def __next__(self):
		if self.ptr is None:
			raise StopIteration()  # 引发异常
		else:
			data = self.ptr.data
			self.ptr = self.ptr.next
			return data
	def find(self,val): #查找元素val，找到返回迭代器，找不到返回None
		ptr = self._head.next
		while ptr is not None:
			if ptr.data == val:
				return DoubleLinkList._Iterator(ptr)
			ptr = ptr.next
		return self.end()
```

### 链表和顺序表的选择

**顺序表**

中间插入太慢



**链表**

访问第i个元素太慢

顺序访问也慢(现代计算机有cache，访问连续内存域比跳着访问内存区域快很多)

还多费空间



结论：**尽量选用顺序表。比如栈和队列，都没必要用链表实现**

基本只有在找到一个位置后**反复要在该位置周围进行增删**，才适合用链表

实际工作中几乎用不到链表

### 队列和栈

对着题目模拟吧

```python
class Queue:                                           #####队列的实现方法##########
	_initC = 8		#存放队列的列表的初始容量
	_expandFactor = 1.5  #扩充容量时容量增加的倍数
	def __init__(self):
		self._q = [None for i in range(Queue._initC)]
		self._size = 0				    #队列元素个数
		self._capacity = Queue._initC #队列最大容量
		self._head = self._rear = 0
	def isEmpty(self):
		return self._size == 0
	def front(self):  #看队头元素。空队列导致re
		if self._size == 0:
			raise Exception("Queue is empty")
		return self._q[self._head]
	def back(self):   #看队尾元素，空队列导致re
		if self._size == 0:
			raise Exception("Queue is empty")
		if self._rear > 0:
			return self._q[self._rear - 1]
		else:
			return self._q[-1]
    def push(self,x):
		if self._size == self._capacity:
			tmp = [None for i in range(
					int(self._capacity*Queue._expandFactor))]
			k = 0
			while k < self._size:
				tmp[k] = self._q[self._head]
				self._head = (self._head + 1) % self._capacity
				k += 1
			self._q = tmp  #原来self._q的空间会被Python自动释放
			self._q[k] = x
			self._head,self._rear = 0,k+1
			self._capacity = int(
					self._capacity*Queue._expandFactor)
		else:
			self._q[self._rear] = x
			self._rear = (self._rear + 1) % self._capacity
		self._size += 1
	def pop(self):
		if self._size == 0:
			raise Exception("Queue is empty")
		self._size -= 1
		self._head = (self._head + 1) % len(self._q)
```

## 二叉树

#### **定义**

二叉树是有限个元素的集合。

空集合是一个二叉树，称为空二叉树。

一个元素(称其为“根”或“根结点”)，加上一个被称为“左子树”的二叉树，和一个被称为“右子树”的二叉树，就能形成一个新的二叉树。要求根、左子树和右子树三者没有公共元素。

#### **相关概念**

二叉树的的元素称为“结点”。结点由三部分组成：数据、左子结点指针、右子结点指针。

**结点的度**(degree)：结点的非空子树数目。也可以说是结点的**子结点数目**。

**叶结点**(leaf node)：度为0的结点。

**分支结点**：度不为0的结点。即除叶子以外的其他结点。也叫内部结点。

**兄弟结点**(sibling)：父结点相同的两个结点，互为兄弟结点。

**结点的层次**(level)：树根是第0层的。如果一个结点是第n层的，则其子结点就是第n+1层的。

**结点的深度**(depth)：即结点的层次。

**祖先**(ancestor):

1) 父结点是子结点的祖先
2) 若a是b的祖先，b是c的祖先，则a是c的祖先。

**子孙**(descendant)：也叫后代。若结点a是结点b的祖先，则结点b就是结点a的后代。

**边**：若a是b的父结点，则对子<a,b>就是a到b的边。在图上表现为连接父结点和子结点之间的线段。

二叉树的**高度**(height)：二叉树的高度就是结点的最大层次数。只有一个结点的二叉树，高度是0。结点一共有n层，高度就是n-1。

**完美二叉树(**perfect binary tree)：每一层结点数目都达到最大。即第i层有2i个结点。高为h的完美二叉树，有2h+1 -1个结点

**满二叉树（**full binary tree)：没有1度结点的二叉树

**完全二叉树**(complete binary tree)除最后一层外，其余层的结点数目均达到最大。而且，最后一层结点若不满，则缺的结点定是在最右边的连续若干个

#### 性质

1) 第i层最个多**2^i**个结点

2) 高为h的二叉树结点总数最多**2^(h+1)-1**

3) 结点数为n的树，边的数目为n-1
4) n个结点的非空二叉树至少有⌈log2(n+1)⌉层结点，即高度至少为⌈log2(n+1)⌉- 1

5) **在任意一棵二叉树中，若叶子结点的个数为n0，度为2的结点个数为n2，则n0=n2+1。**

6) 非空满二叉树叶结点数目等于分支结点数目加1。
7) 非空二叉树中的空子树数目等于其结点数目加1

#### 完全二叉树的性质

1) 完全二叉树中的1度结点数目为0个或1个

2) 有n个结点的完全二叉树有⌊(n+1)/2⌋个叶结点。

3) 有n个叶结点的完全二叉树有2n或2n-1个结点(两种都可以构建)

4) 有n个结点的非空完全二叉树的高度为⌈log2(n+1)⌉-1。即：有n个结点的非空完全二叉树共有⌈log2(n+1)⌉层结点。

#### 最优二叉树（Huffman编码树）

1)开始n个结点位于集合S

2)从S中取走两个权值最小的结点n1和n2，构造一棵二叉树，树根为结点r，r的两个子结点是n1和n2，且Wr=Wn1+Wn2，并将r加入S

3)重复2）,直到S中只有一个结点，最优二叉树就构造完毕，根就是S中的唯一结点

显然，最优二叉树不唯一

**Huffman:**

**基本思想：使用频率越高的字符，离树根越近。构造过程和最优二叉树一样**

过程:

**1.** 开始时，若有n个字符，则就有n个结点。每个结点的**权值**就是字符的频率，每个结点的**字符集**就是一个字符。

**2.** 取出权值最小的两个结点，合并为一棵子树。子树的树根的权值为两个结点的权值之和，字符集为两个结点字符集之并。在结点集合中删除取出的两个结点，加入新生成的树根。

**3.** 如果结点集合中只有一个结点，则建树结束。否则，goto **2**

## 堆

1)堆(二叉堆)是一个**完全二叉树**

2)堆中任何结点优先级都高于或等于其两个子结点（什么叫优先级高可以自己定义）

3)一般将**堆顶元素最大**的堆称为大根堆（大顶堆），**堆顶元素最小**的堆称为小根堆（小顶堆)

存储：用**列表**存放堆。堆顶元素下标是0。下标为i的结点，其左右子结点下标分别为2i+1,2i+2。

#### 性质

1)堆顶元素是优先级最高的(啥叫优先级高可自定义）

2)堆中的任何一棵子树都是堆

3)往堆中添加一个元素，并维持堆性质，复杂度**O(log(n))**

4)删除堆顶元素，剩余元素依然维持堆性质，复杂度**O(log(n))**

5)在无序列表中原地建堆，复杂度**O(n)**

#### 应用

堆用于需要经常从一个集合中取走(即删除)优先级最高元素，而且还要经常往集合中添加元素的场合(堆可以用来实现优先队列）

可以用堆进行排序，复杂度**O(nlog(n))**，且只需要O(1)的额外空间,称为“堆排序”。递归写法需要o(log(n))额外空间，非递归写法需要O(1)额外空间。

#### 操作

**添加：**

1)假设堆存放在列表a中，长度为n

2)添加元素x到列表a尾部，使其成为a[n]

3)若x优先级高于其父结点，则令其和父结点交换，直到x优先级不高于其父结点，或x被交换到a[0]，变成堆顶为止。此过程称为将x"上移"

4)x停止交换后，新的堆形成，长度为n+1

**复杂度O(log(n))**

**删除堆顶元素：**

1)假设堆存放在列表a中，长度为n

2)将a[0]和a[n-1]交换

3)将a[n-1]（刚才的a[0]）删除(pop)

4)记此时的a[0]（刚才的a[n-1]）为x，则将x和它两个儿子中优先级较高的，且优先级高于x的那个交换，直到x变成叶子结点，或者x的儿子优先级都不高于x为止。将此整个过程称为将x"下移"

5)x停止交换后，新的堆形成，长度为n-1

**复杂度O(log(n))**

**建堆：复杂度O(n)**

#### 实现

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        print(f'i = {i}, {self.heapList}')



bh = BinHeap()
bh.buildHeap([9, 5, 6, 2, 3])
"""
i = 2, [0, 9, 5, 6, 2, 3]
i = 1, [0, 9, 2, 6, 5, 3]
i = 0, [0, 2, 3, 6, 5, 9]
"""

for _ in range(bh.currentSize):
    print(bh.delMin())
"""
2
3
5
6
9
""
```

## 树

#### 概念

每个结点可以有任意多棵不相交的子树

子树有序，从左到右依次是子树1,子树2......

二叉树的结点在只有一棵子树的情况下，要区分是左子树还是右子树。树的结点在只有一棵子树的情况下，都算其是第1棵子树（所以二叉树不是树)

支持广度优先遍历、前序遍历(先处理根结点，再依次处理各个子树）和后序遍历（先依次处理各个子树，处理根结点），中序遍历无明确定义

#### 性质

结点度数最多为K的树，第i层最多K^i个结点(i从0开始）。

结点度数最多为K的树，高为h时最多有(K^(h+1) -1 )/(k-1) 个结点。

n个结点的K度完全树，高度h是logk (n)向下取整

n个结点的树有n-1条边

## 森林

不相交的树的集合，就是森林

森林有序，有第1棵树、第2棵树、第3棵树之分

森林可以表示为树的列表，也可以表示为一棵二叉树

1) 森林中第1棵树的根，就是二叉树的根S1，S1及其左子树，是森林的第1棵树的二叉树表示形式

2) S1的右子节S2，以及S2的左子树，是森林的第2棵树的二叉树表示形式

3) S2的右子节S3，以及S3的左子树，是森林的第3棵树的二叉树表示形式

#### 森林→二叉树

```python
def woodsToBinaryTree(woods):
	#woods是个列表，每个元素都是一棵二叉树形式的树
	
	biTree = woods[0]
	p = biTree
	for i in range(1,len(woods)):
		p.addRight(woods[i])
		p = p.right
	return biTree
#biTree和woods共用结点,执行完后woods的元素不再是原儿子兄弟树
```

#### 二叉树→森林

```python
def binaryTreeToWoods(tree):
#tree是以二叉树形式表示的森林
	p = tree
	q = p.right
	p.right = None
	woods = [p]
	if q:
		woods += binaryTreeToWoods(q)
	return woods

##woods是兄弟-儿子树的列表,woods和tree共用结点 
##执行完后tree的元素不再原儿子兄弟树
```

## 二叉排序（搜索）树 BST

是一棵二叉树

每个结点存储关键字(key)和值(value)两部分数据

对每个结点X，其左子树中的全部结点的key都小于X的key，且X的key小于其右子树中的全部结点的key一个二叉搜索树中的任意一棵子树都是二叉搜索树

性质：一个二叉树是二叉搜索树，当且仅当其中序遍历序列是递增序列

略作修改就可以处理树结点key可以重复的情况。

#### 查找

递归过程，查找key为X的结点的value

如果X和根结点相等，则返回根结点的value，查找结束

如果X比根结点小，则递归进入左子树查找

如果X比根结点大，则递归进入右子树查找

#### **插入**

递归过程，插入key为X的结点

如果X和根结点相等，则更改根结点的value

如果X比根结点小，则递归插入到左子树。如果没有左子树，则新建左子结点，存放要插入的key和value，插入工作结束。

如果X比根结点大，则递归插入到右子树。如果没有右子树，则新建右子结点，存放要插入的key和value，插入工作结束。

#### **删除**

**二叉排序树建树复杂度可以认为是O(nlog(n))。平均情况下，建好的二叉排序树深度是log(n)**

```python
class BinarySearchTree:
    class Node:  # 结点类
        def __init__(self, key, data, left=None, right=None):
            self.key, self.value, self.left, self.right = \
               key, data, left, right
    
	def __init__(self,less=lambda x,y:x<y ):
        self.root,self.size = None,0 #root是树根，size是结点总数
        self.less = less 			#less是比较函数
	def _find(self,key): #查找值为key的结点,返回值是找到的结点及其父亲
        def find(root,father): #在以root为根的子树中查找
            #father 是root的父亲，返回找到的结点及其父亲
            if self.less(key, root.key):
                if root.left:
                    return find(root.left, root)
                else:
                    return None,None  #找不到
            elif self.less(root.key, key):
                if root.right:
                    return find(root.right,root)
                else:
                    return None,None
            else:
                return root, father
        if self.root is None:
            return None,None
        return find(self.root,None)
	def _insert(self,key,data): #插入结点(key,data)
        def insert(root): #返回值表示是否插入了新结点
            if self.less(key, root.key):
                if root.left is None:
                    root.left = BinarySearchTree.Node(key, data)
                    return True #插入了新结点
                else:
                    return insert( root.left)
            elif self.less(root.key, key):
                if root.right is None:
                    root.right = BinarySearchTree.Node(key, data)
                    return True
                else:
                    return insert(root.right)
            else:
                root.value = data  # 相同关键字，则更新
                return False
        if self.root is None:
            self.root = BinarySearchTree.Node(key,data)
            self.size = 1
        else:
            self.size += insert(self.root)

    def _findMin(self,root,father): #找以root为根的子树的最小结点及其父亲
        #father是root的父亲
        if root.left is None:
            return root,father
        else:
            return self._findMin(root.left,root)
    def _findMax(self,root,father): #找以root为根的子树的最大结点及其父亲
        if root.right is None:
            return root,father
        else:
            return self._findMax(root.right,father)
    def pop(self,key):
        #删除键为key的结点，返回该结点的data。如果没有这样的元素，则引发异常
        nd,father = self._find(key)
        if nd is None:
            raise Exception("key not found")
        else:
            self.size -= 1
            self._deleteNode(nd,father)
            return nd.value
    def _deleteNode(self,nd,father): #删除结点nd,nd的父结点是father
        if nd.left and nd.right: #nd左右子树都有
            minNd,father = self._findMin(nd.right,nd)
            nd.key,nd.value = minNd.key,minNd.value
            self._deleteNode(minNd,father)
        elif nd.left:	#nd只有左子树
            if father and father.left is nd: #nd是父亲的左儿子
                father.left = nd.left
            elif father  and father.right is nd:
                father.right = nd.left
            else: #nd是树根
                self.root = nd.left
        elif nd.right : #nd只有右子树
            if father and father.right is nd:
                father.right = nd.right
            elif father and father.left is nd:
                father.left = nd.right
            else:
                self.root = nd.right
        else: #nd是叶子
            if father and father.left is nd:
                father.left = None
            elif father and father.right is nd:
                father.right = None
            else: #nd是树根
                self.root = None

    def _inorderTraversal(self): 
        def inorderTraversal(root): 
            if root.left:
                yield from inorderTraversal(root.left)
            yield root.key,root.value
            if root.right:
                yield from inorderTraversal(root.right)
        if self.root is None:
            return
        yield from inorderTraversal(self.root)
    def __contains__(self, key): #实现 in
        return self._find(key)[0] is not None
    def __iter__(self): #返回迭代器
        return self._inorderTraversal()
    def __getitem__(self,key):  #实现右值 []
        nd,father = self._find(key)
        if nd is None:
            raise Exception("key not found")
        else:
            return nd.value
    def __setitem__(self, key, data): #实现左值 []
        nd,father = self._find(key)
        if nd is None:
            self._insert(key,data)
        else:
            nd.value = data
    def __len__(self):
        return self.size
import random
random.seed(2)
s = [i for i in range(8)] 
tree = BinarySearchTree()  
#若 tree = Tree(lambda x ,y : y <x) 则从大到小排
random.shuffle(s)
for x in s:
    tree[x] = x		#加入关键字为x，值为x的元素
print(len(tree))    #>>8
for x in tree:  #首先会调用tree.__iter__()返回一个迭代器
    print(f"({x[0]},{x[1]})",end = "") #从小到大遍历整个树
#>>(0,0)(1,1)(2,2)(3,3)(4,4)(5,5)(6,6)(7,7)
print()
print(3000 in tree)	#>>False
print(3 in tree)    	#>>True
print(tree[3])      	#>>3  输出关键字为3的元素的值
try:
    print(tree[3000])  #关键字为3000的元素不存在，此句引发异常
except Exception as e:
    print(e)			#>>key not found
tree[3000] = "ok"	#添加关键字为3000，值为"ok"的元素
print(tree[3000],len(tree))		#>>ok 9
tree[3000] = "bad"	#将关键字为3000的元素的值改为"bad"
print(tree[3000],len(tree))		#>>bad 9
try:
    tree.pop(354)		#关键字为354的元素不存在，此句引发异常
except Exception as e:
    print(e)				#>>key not found
tree.pop(3)
print(len(tree))        #>>8
```

## 散列表

如果能在元素的存储位置和其关键字之间建立某种直接关系，那么在进行查找时，就无需做比较或做很少次的比较，按照这种关系直接由关键字找到相应的记录。这就是散**列查找法**（Hash Search）的思想，它通过对元素的关键字值进行某种运算，直接求出元素的地址，即使用关键字到地址的直接转换方法，而不需要反复比较。因此，散列查找法又叫杂凑法或散列法。

下面给出散列法中常用的几个术语。

(1) **散列函数和散列地址**：在记录的存储位置p和其关键字 key 之间建立一个确定的对应关系 H，使p=H(key)，称这个对应关系H为散列函数，p为散列地址。

(2) **散列表**：一个**有限连续**的地址空间，用以存储按散列函数计算得到相应散列地址的数据记录。通常散列表的存储空间是一个一维数组，散列地址是数组的下标。

(3) **冲突和同义词**：对不同的关键字可能得到同一散列地址,即 key1≠key2,而 H(key1) = H(key2) 这种现象称为冲突。具有相同函数值的关键字对该散列函数来说称作同义词，key1与 key2 互称为同义词。

#### 常用：除留余数法

假设散列表表长为 m，选择一个不大于m 的数p，用p去除关键字，除后所得余数为散列地址，即 H(key) = key%p

这个方法的关键是选取适当的p，一般情况下，可以选p为小于表长的最大**质数**。例如，表长m=100，可取p=97。

除留余数法计算简单，适用范围非常广，是最常用的构造散列函数的方法。它不仅可以对关键字直接取模，也可在折叠、平方取中等运算之后取模，这样能够保证散列地址一定落在散列表的地址空间中。

**处理冲突**：

1. 使用第一个散列函数 `hash1(key)` 计算关键字 `key` 的初始散列值 `hash_value = hash1(key)`。
2. 如果散列表中的槽位 `hash_value` 是空的，则将关键字 `key` 插入到该槽位中。
3. 如果槽位 `hash_value` 不为空，表示发生了冲突。在这种情况下，我们**使用第二个散列函数 `hash2(key)` 来计算关键字 `key` 的步长（step）**。
4. 通过计算 `step = hash2(key)`，我们将跳过 `step` 个槽位，继续在散列表中查找下一个槽位。
5. 重复步骤 3 和步骤 4，直到找到一个空槽位，将关键字 `key` 插入到该槽位中。

# 图

图由顶点集合和边集合组成每条边连接两个不同顶点

有向图：边有方向(有起点和终点）

无向图：边没有方向

边只是逻辑上表示两个顶点有直接关系，边是直的还是弯的，边有没有交叉，都没有意义。

无向图两个顶点之间**最多一条**边

有向图两个顶点之间最多**两条方向不同**的边

1)顶点的度数：和顶点相连的边的数目。

2)顶点的出度：有向图中，以该顶点作为起点的边的数目

3)顶点的入度：有向图中，以该顶点作为终点的边的数目

4)顶点的出边：有向图中，以该顶点为起点的边

5)顶点的入边：有向图中，以该顶点为终点的边

6)路径：对于无向图，如果存在顶点序列Vi0,Vi1Vi2.....Vim,使得(Vi0,Vi1) ,(Vi1,vi2)...(Vim-1,Vim)都存在，则称（Vi0,Vi1...Vim)是从Vi0到Vim的一条路径。（对于有向图，把()换成<>)

7)路径的长度：路径上的边的数目

8)回路（环）：起点和终点相同的路径

9)简单路径：除了起点和终点可能相同外，其它顶点都不相同的路径

**完全图：**

完全无向图：任意两个顶点**都有边相连**

完全有向图：任意两个顶点都有**两条方向相反的边**

9)连通：如果存在从顶点u到顶点v的路径，则称u到v连通，或u可达v。无向图中，u可达v,必然v可达u。有向图中，u可达v，并不能说明v可达u。

10)连通无向图：图中任意两个顶点u和v互相可达。

11)强连通有向图：图中任意两个顶点u和v**互相可达**。

12)子图：从图中抽取部分或全部边和点构成的图

13)连通分量（极大连通子图）：无向图的一个子图，是连通的，且再添加任何一些原图中的顶点和边，新子图都不再连通。

14)强连通分量：**有向图**的一个**子图**，是**强连通**的，且再添加任何一些原图中的顶点和边，新子图都不再强连通。

15)带权图：边被赋予一个权值的图

16)网络：带权无向连通图

#### **性质**：

1.图的边数等于顶点度数之和的一半

2.n个顶点的**连通图**至少有n-1条边

3.n个顶点的，**无回路**的连通图就是一棵树，有n-1条边

#### 表示方法：

邻接矩阵 or 邻接表

### 拓扑排序

拓扑排序（Topological Sorting）：在有向图中求一个顶点的序列，使其满足以下条件:

  1）每个顶点出现且只出现一次

  2）若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面

有向图或AOV网存在拓扑排序的充要条件：

  图中**无环**，即图是有向无环图DAG( Directed Acyclic Graph）

#### 实现：

1. 从图中任选一个没有前驱（入度为0）的顶点 x 输出

2. 从图中删除 x 和所有以它为起点的边

重复 1 和 2 直到图为空或当前图中不存在无前驱的顶点为止(后一种情况说明图中有环，无法拓扑排序)

具体实现：用队列存放入度变为0的点。每个顶点出入队列一次，每个顶点连的边都要看一次，复杂度O(E+V)

```python
class Edge:
	def __init__(self,v,w):
		self.v,self.w = v,w  #v是顶点，w是权值
def topoSort(G): #G是邻接表，顶点从0开始编号
	#G[i][j]是Edge对象
	n = len(G)
	import queue
	inDegree = [0] * n #inDegree[i]是顶点i的入度
	q = queue.Queue()
	for i in range(n):
		for e in G[i]:
			inDegree[e.v] += 1
	for i in range(n):
		if inDegree[i] == 0:
			q.put(i)
	seq = []
	while not q.empty():
		k = q.get()
		seq.append(k)
		for e in G[k]:
			inDegree[e.v] -= 1  #删除边(k,v)后将v入度减1
			if inDegree[e.v] == 0:
				q.put(e.v)
	if len(seq) != n:  #如果拓扑序列长度少于点数，则说明有环
		return None
	else:
		return seq
```

## 最小生成树

#### 生成树：

在一个无向连通图G中，如果取它的全部顶点和一部分边构成一个子图G’，即：

   V(G’)=V(G);E(G’) ⊆E(G)

 若边集E(G’)中的边既将图中的所有顶点连通又不形成回路，则称子图G’是原图G的一棵生成树。

一棵含有n个点的生成树，**必含有n-1条边**。

无向图的**极小连通子图**(去掉一条边就不连通的子图)就是生成树

#### 最小生成树：

对于一个无向连通带权图，每棵树的**权**（即树中所有**边的权值总和**）也可能不同

具有**最小权值**的生成树称为最小生成树。

### Prim算法

假设G=(V,E)是有n个顶点的带权连通图，T=(U,TE)是G的最小生成树，U,TE初值均为空集。

从V中任取一个顶点将它并入U中

每次从一个端点已在T中，另一个端点仍在T外的所有边中，找一条权值最小的，并把该边及端点并入T。做n-1次，T中就有n个点，n-1条边，T就是最小生成树

```python
import heapq
INF = 1 << 30

class Edge:
    def __init__(self,v=0,w=INF):
        self.v = v  #边端点，另一端点已知
        self.w = w  #边权值或v到在建最小生成树的距离
    def __lt__(self,other):
        return self.w < other.w
def HeapPrim(G,n):
    #G是邻接表,n是顶点数目，返回值是最小生成树权值和
    xDist = Edge(0,0)
    pq = []
    heapq.heapify(pq) #存放顶点及其到在建生成树的距离
    vDist = [INF for i in range(n)] #各顶点到已经建好的那部分树的距离
    vUsed = [0 for i in range(n)] #标记顶点是否已经被加入最小生成树
    doneNum = 0  #已经被加入最小生成树的顶点数目
    totalW = 0 #最小生成树总权值
    heapq.heappush(pq,Edge(0,0)) #开始只有顶点0，它到最小生成树距离0
    while doneNum < n and pq!= []:
        while True:  #每次从堆里面拿离在建生成树最近的不在生成树里面的点
            xDist = pq[0]
            heapq.heappop(pq)
            if not (vUsed[xDist.v] == 1 and  pq != []):
                break
        if vUsed[xDist.v] == 0: #xDist.v要新加到生成树里面
              totalW += xDist.w
              vUsed[xDist.v] = 1
              doneNum +=1
              for i in range(len(G[xDist.v])): #更新新加入点的邻点
                    k = G[xDist.v][i].v
                    if vUsed[k] == 0:
                            w = G[xDist.v][i].w
                            if  vDist[k] > w:
                                   vDist[k] = w
                                   heapq.heappush(pq,(Edge(k,w)))
    if doneNum < n:
        return -1; #图不连通
    return totalW
```

### Kruskal算法

假设G=(V,E)是一个具有n个顶点的连通网，T=(U,TE)是G的最小生成树，U=V,TE初值为空。

将图G中的边按权值从小到大依次选取，若选取的边使生成树不形成回路，则把它并入TE中，若形成回路则将其舍弃，直到TE中包含N-1条边为止，此时T为最小生成树。

```python
INF = 1 << 30
class Edge:
    def __init__(self,s,e,w):
        self.s ,self.e,self.w = s,e,w  #起点，终点，权值
    def __lt__(self,other):
        return self.w < other.w

def GetRoot(a):
	if parent[a] == a:
		return a
	parent[a] = GetRoot(parent[a])
	return parent[a]
def Merge(a, b):
	p1 = GetRoot(a)
	p2 = GetRoot(b)
	if p1 == p2: 	
		return
	parent[p2] = p1
while True:  #main
    try:
        N = int(input())
        parent = [i for i in range(N)]
        edges = []
        for  i in range(N):
            lst = list(map(int,input().split()))
            for j in range(N):
                edges.append(Edge(i,j,lst[j]))
        edges.sort() #排序复杂度O(ElogE）
        done = totalLen = 0
        for edge in edges:
            if GetRoot(edge.s) != GetRoot(edge.e):
                Merge(edge.s,edge.e)
                done += 1
                totalLen += edge.w
            if done == N - 1:
                break
        print(totalLen)
    except:  break
```

#### 比较：

•**Kruskal**:将所有边从小到大加入，在此过程中判断是否构成回路

–使用数据结构：并查集

–时间复杂度：O(ElogE)

–适用于稀疏图

•**Prim**:从任一节点出发，不断扩展

–使用数据结构：堆

–时间复杂度：O(ElogV) 或 O(VlogV+E)(斐波那契堆）

–适用于密集图

–若不用堆则时间复杂度为O(V2)

## 最短路

### Dijkstra(无负权值)

用邻接表，不优化，时间复杂度O(V2+E)

Dijkstra+堆的时间复杂度 o(ElgV)

用斐波那契堆可以做到O(VlogV+E)

```python
import heapq
class Edge:
    def __init__(self,k=0,w=0):
        self.k ,self.w = k,w  #有向边的终点和边权值，或当前k到源点的距离
    def __lt__(self,other):
        return self.w < other.w

bUsed = [0 for i in range(30010)]# bUsed[i]为1表示源到i的最短路已经求出
INF = 100000000
N,M = map(int,input().split())
G = [[] for i in range(N+1)]
for i in range(M):
    s,e,w = map(int,input().split())
    G[s].append(Edge(e,w))
pq = []
heapq.heapify(pq)
heapq.heappush(pq,Edge(1,0)) #源点是1号点,1号点到自己的距离是0
while pq != []:
    p = pq[0]
    heapq.heappop(pq)
    if bUsed[p.k]: #已经求出了最短路
        continue
    bUsed[p.k] = 1
    if p.k == N:  #因只要求1-N的最短路，所以要break
        break
    L = len(G[p.k])
    for i in range(L):
        q = Edge()
        q.k = G[p.k][i].k
        if bUsed[q.k]:
            continue
        q.w = p.w + G[p.k][i].w
        heapq.heappush(pq,q) #队列里面已经有q.k点也没关系
print( p.w )
```

### Floyd算法

•用于求每一对顶点之间的最短路径。有向图，无向图均可。有向图可以有负权边，但是不能有负权回路。

•假设求从顶点vi到vj的最短路径。如果从vi到vj有边，则从vi到vj存在一条长度为cost[i,j]的路径，该路径不一定是最短路径，尚需进行n次试探。

•考虑路径（ vi， v1 ，vj）是否存在（即判别弧（ vi， v1 ）和（ v1 ，vj ）是否存在）。如果存在，则比较cost[i,j]和（ vi， v1 ，vj）的路径长度，取长度较短者为从vi到vj的中间顶点的序号不大于1的最短路径，记为新的cost[i,j] 。

假如在路径上再增加一个顶点v2 ，如果（ vi，**…**， v2 ）和（ v2 ，**…**，vj ）分别是当前找到的中间顶点的序号不大于2的最短路径，那么（ vi，**…****，** v2 ，**…** ， vj ）就有可能是从vi到 vj的中间顶点的序号不大于2的最短路径。将它和已经得到的从vi到 vj的中间顶点的序号不大于1的最短路径相比较，从中选出中间顶点的序号不大于2的最短路径之后，再增加一个顶点v3 ，继续进行试探。依次类推

•复杂度O(n^3)。不能处理带负权边的无向图，和有负权回路的有向图

•记distk(i,j)为从Vi到Vj的途经的顶点编号不大于k的最短路长度，则有：

 dist^-1(i,j) = Wi,j(Wi,j是边(i,j)权值，边不存在则为无穷大）

 dist^0(i,j) = min{dist-1(i,j), dist-1(i,0) + dist-1(0,j)

 dist^1(i,j) = min{dist0(i,j), dist0(i,1) + dist0(1,j)

 dist^k(i,j) = min{ distk-1(i,j), distk-1(i,k) + distk-1(k,j)} 

 dist^(n-1)(i,j) = min{ distn-2(i,j), distn-2(i,n-1) + distn-2(n-1,j)}  

其中dist-1(i,j)表示从Vi到Vj的不途经任何顶点的最短路径长度。

dist^(n-1)(i,j)就是Vi到Vj的最短路的长度

```python
def  floyd(G): #G是邻接矩阵，顶点编号从0开始算,无边则边权值为INF
	n = len(G)
	INF = 10**9
	prev = [[None for i in range(n)] for j in range(n)]
   #prev[i][j]表示到目前为止发现的从i到j的最短路上，j的前驱。
	dist = [[INF for i in range(n)] for j in range(n)]
	for i in range(n):
		for j in range(n):
			if i == j:
				dist[i][j] = 0
			else:
				if G[i][j] != INF: #i到j的边存在
					dist[i][j] = G[i][j]
					prev[i][j] = i
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if dist[i][k] + dist[k][j] < dist[i][j]:
					dist[i][j] = dist[i][k] + dist[k][j]
					prev[i][j] = prev[k][j]
	return dist,prev
```

## 内排序

在内存中进行的排序，叫内排序，简称排序

复杂度不可能优于**O(nlog(n))**

对外存（硬盘）上的数据进些排序，叫外排序

**时间复杂度**

  平均复杂度

  最坏情况复杂度

  最好情况复杂度

**空间复杂度**

  需要多少额外辅助空间

**是否稳定：**

  同样大小的元素，排序前和排序后是否先后次序不变

![image-20240618100424581](C:\Users\Dell\AppData\Roaming\Typora\typora-user-images\image-20240618100424581.png)

#### Timsort，蒂姆排序（python 内置）

一种混合了归并排序和插入排序的算法

稳定

最坏时间复杂度O(nlog(n))

最好时间复杂度接近O(n)

额外空间：最坏O(n)，但通常较少

是目前为止最快的排序算法

#### 插入排序

1）将序列分成有序的部分和无序的部分。有序的部分在左边，无序的部分在右边。开始有序部分只有1个元素

2） 每次找到无序部分的最左元素（设下标为i)，将其插入到有序部分的合适位置(设下标为k,则原下标为k到i-1的元素都右移一位)，有序部分元素个数+1

3） 直到全部有序

```python
def insertionSort(a):
	for i in range(1,len(a)): #每次把a[i]插入到合适位置
		e,j = a[i],i
		while j > 0 and e < a[j-1]: #比写 a[j-1]>e 适应性强
			a[j] = a[j-1]		#(1)
			j -= 1
		a[j] = e
```

最坏情况(倒序）: 语句(1）执行1+2+3+...+(n-1)次，总复杂度O(n^2)

平均情况: 对每个i, 语句（1）平均执行i/2次，总复杂度O(n^2)

最好情况（基本有序): 对每个i,语句(1)不执行或执行很少次，总复杂度O(n)

稳定性：**稳定**

额外空间：O(1)

可以用二分法或自定义比较器来改进

**规模很小**的排序可优先选用(比如，元素个数10以内)

特别适合元素**基本有序**的情况(复杂度接近O(n))

许多算法会在上述两种情况下采用插入排序。例如改进的快速排序算法、归并排序算法，在待排序区间很小的时候就不再递归快排或归并，而是用插入排序

#### 希尔排序（改进的插入排序）

1)选取增量(间隔)为D，根据增量将列表分为多组，每组分别插入排序:

   第一组：A0 , A0+D , A0+2D, ......

   第二组：A1 , A1+D , A1+2D, ......

   第三组：A2 , A2+D , A2+2D, ......

   若D==1，则插入排序后，整个排序结束 

2)D = D//2 ，转1 

初始增量D可以为 n//2, n是元素总数

也许D还可以有别的选取法

**最好：O(n)，平均O(n^1.5)，最坏O(n^2)**

#### 选择排序

1） 将序列分成有序的部分和无序的部分。有序的部分在左边，无序的部分在右边。开始有序部分没有元素

2） 每次找到无序部分的**最小元素**（设下标为i) ，和无序部分的**最左边元素**（设下标为j)**交换**。有序部分元素个数+1。

3） 做n-1次，排序即完成

```python
def selectionSort(a):
	n = len(a)
	for i in range(n-1):
		minPos = i #最小元素位置
		for j in range(i+1,n):
			if a[j] < a[minPos]: #(1)
				minPos = j
		if minPos != i:
			a[minPos],a[i] = a[i],a[minPos]
```

无论最好、最坏、平均，语句(1）必定执行(n-1)+...+3+2+1次，复杂度**O(n^2)**

稳定性：**不稳定**，因a[i]被交换时，可能越过了其后面一些和它相等的元素

额外空间：O(1)

平均效率低于插入排序，**没啥实际用处**

#### **冒泡排序**

1） 将序列分成有序的部分和无序的部分。有序的部分在**右**边，无序的部分在**左**边。开始有序部分没有元素

2） 每次从左到右，依次比较无序部分相邻的两个元素。如果右边的小于左边的，则交换它们。做完一次后，**无序部分最大元素 即被换到无序部分最右边**，有序部分元素个数+1。

3） 做n-1次，排序即完成

```python
def bubbleSort(a):
	n = len(a)
	for i in range(1,n):
		for j in range(n-i):
			if a[j+1] < a[j]: #(1)
				a[j+1],a[j] = a[j],a[j+1]
```

无论最好、最坏、平均，语句(1）必定执行(n-1)+...+3+2+1次，复杂度**O(n^2)**

稳定性：**稳定**

额外空间：O(1)

**改进**：最好情况，即**基本有序**时，可以做到**O(n)**

如果发现某一轮扫描时，没有发生元素交换的情况，则说明已经排好序了，就不用再扫描了

```python
def bubbleSort(a):
	n = len(a)
	for i in range(1,n):
		done = True
		for j in range(n-i):
			if a[j+1] < a[j]: #(1)
				a[j+1],a[j] = a[j],a[j+1]
				done = False
		if done:
			break
```

#### 归并排序

1. 把前一半排序

2) 把后一半排序
3) 把两半归并到一个新的有序数组，然后再拷贝回原数组，排序完成。

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves

		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half

		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1

if __name__ == '__main__':
	arr = [12, 11, 13, 5, 6, 7]
	mergeSort(arr)
	print(' '.join(map(str, arr)))
# Output: 5 6 7 11 12 13
```

无所谓最坏、最好或平均情况，复杂度都是: **O(nlog(n))**

稳定性：**稳定**。只要归并两个区间时，碰到相同元素，总是先取左区间元素即可

额外空间：**O(n)**

归并用额外空间O(n) + 栈空间O(log(n))

可用来求逆序数

#### 快速排序

1. 设k=a[0], 将k挪到适当位置，使得比k小的元素都在k左边,比k大的元素都在k右边，和k相等的，不关心在k左右出现均可 （O（n)时间完成）

2) 把k左边的部分快速排序
3) 把k右边的部分快速排序

**最坏情况(已经基本有序或倒序): O(n^2)**

平均情况: O(nlog(n))

最好情况: O(nlog(n))

稳定性：**不稳定**

额外空间：

两次递归的普通写法：**最坏情况**需要递归n层，需要n层栈空间，复杂度**O(n)**。**最好情况和平均情况**递归log(n)层，复杂度**O(log(n))**

如何**避免**最坏情况的发生

 办法1) 排序前O(n)随机打乱

 办法2) 若待排序段为a[s,e]，则选a[s],a[e],a[(s+e)/2]三者中的中位数作为分隔基准元素

#### 堆排序

1)将待排序列表a变成一个堆(O(n))

2)将a[0]和a[n-1]交换，然后对新a[0]做下移，维持前n-1个元素依然是堆。此时优先级最高的元素就是a[n-1]

3)将a[0]和a[n-2]交换，然后对新a[0]做下移,  维持前n-2个元素依然是堆。此时优先级次高的元素就是a[n-2]

......

直到堆的长度变为1，列表a就按照优先级从低到高排好序了。

整个过程相当不断删除堆顶元素放到a的后部。堆顶元素依次是优先级最高的、次高的....

一共要做n次下移，每次下移O(log(n))，因此总复杂度**O(nlog(n))**

如果用递归实现，需要**O(log(n))**额外栈空间(递归要进行log(n)层)。

如果不用递归实现，需要**O(1)**额外空间。

稳定性：**不稳定**

```python
import heapq
def heapSorted(iterable): #iterable是个序列
#函数返回一个列表，内容是iterable中元素排序的结果，不会改变iterable
	h = []
	for value in iterable:
		h.append(value)
	heapq.heapify(h)  #将h变成一个堆
	return [heapq.heappop(h) for i in range(len(h))]
```

```python
def heapSort(a,key = lambda x:x): #对列表a进行排序
	def makeHeap(): #建堆
		i = (heapSize - 1 - 1) // 2 #i是最后一个叶子的父亲
		for k in range(i,-1,-1):
			shiftDown(k)
	def shiftDown(i): #a[i]下移
		while i * 2 + 1 < heapSize:  #只要a[i]有儿子就做
			L,R = i * 2 + 1, i * 2 + 2
			if R >= heapSize or key(a[L]) < key(a[R]):
				s = L
			else:
				s = R
			if key(a[s]) < key(a[i]):
				a[i],a[s] = a[s],a[i]
				i = s
			else:
				break
	heapSize = len(a)
	makeHeap()
	for i in range(len(a)-1,0,-1):
		a[i],a[0] = a[0],a[i]
		heapSize -= 1
		shiftDown(0)
	n = len(a)
	for i in range(n//2): #颠倒a
		a[i],a[n-1-i] = a[n-1-i],a[i]
```

#### 分配排序(桶排序)

如果待排序元素只有m种不同取值，且m很小（比如考试分数只有0-100),可以采用桶排序

设立m个桶，分别对应m种取值。桶和桶可以比大小，桶的大小就是其对应取值的大小。把元素依次放入其对应的桶，然后再按先小桶后大桶的顺序，将元素都收集起来，即完成排序

复杂度**O(n+m)**，且**稳定**。n是待排序元素个数。

额外空间：**O(n+m)**

例如：将考试分数分到0-100这101个桶里面，然后按照0、1、2...100的顺序收集桶里的分数，即完成排序

```python
def bucketSort(s,m,key=lambda x:x):
    buckets = [[] for i in range(m)]
    for x in s:
        buckets[key(x)].append(x)
    i = 0
    for bkt in buckets:
        for e in bkt:
            s[i] = e
            i += 1
```

#### 多轮分配排序（基数排序）

1)将待排序元素看作由相同个数的原子构成的元组（e1,e2...en)。长度不足的元素，用最小原子补齐左边空缺的部分。

2)原子种类必须很少。有n种原子，就设立n个桶

3)先按en将所有元素分配到各个桶里，然后从小桶到大桶收集所有元素，得到序列1,然后将序列1按en-1分配到各个桶里再收集成为序列2.....直到按e0分配到桶再完成收集得到序列n，序列n就是最终排序结果。

基数排序的复杂度是 **O(d*(n+radix))**

  n : 要排序的元素的个数(假设每个元素由若干个原子组成）

  radix: 桶的个数，即组成元素的原子的种类数

  d:  元素最多由多少个原子组成

  对序列 73,22,93,43,55,14,28,65,39,81 排序: 

  n = 10, d = 2, radix = 10(或9)

 一共要做 d 轮分配和收集

  每一轮, 分配的复杂度 O(n)，收集的复杂度O(radix)

 （一个桶里的元素可以用链表存放，便于快速搜集）

  总复杂度 O(d * ( n + radix))

```python
def radixSort(s, m, d, key):
    #key(x,k)可以取元素x的第k位原子
    for k in range(d):
        buckets = [[] for j in range(m)]
        for x in s:
            buckets[key(x, k)].append(x)
        i = 0
        for bkt in buckets: #这样收集复杂度O(len(s))
            for e in bkt:
                s[i] = e
                i += 1
def getKey(x, i):
    #取非负整数x的第i位。个位是第0位
    tmp = None
    for k in range(i + 1):
        tmp = x % 10
        x //= 10
    return tmp
```

## KMP

对于模式“AAAA”，lps[] 为 [0, 1, 2, 3]

对于模式“ABCDE”，lps[] 为 [0, 0, 0, 0, 0]

对于模式“AABAACAABAA”，lps[] 为 [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]

对于模式“AAACAAAAAC”，lps[] 为 [0, 1, 2, 0, 1, 2, 3, 3, 3, 4]

对于模式“AAABAAA”，lps[] 为 [0, 1, 2, 0, 1, 2, 3]

```python
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """

    m = len(pattern)
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]    # 跳过前面已经比较过的部分
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length
    return lps


def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []

    j = 0  # j是pattern的索引
    for i in range(n):  # i是text的索引
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)
            j = lps[j - 1]
    return matches

text = "ABABABABCABABABABCABABABABC"
pattern = "ABABCABAB"
index = kmp_search(text, pattern)
print("pos matched：", index)
# pos matched： [4, 13]
```

