#include "FerNNClassifier.h"
#include "kern.cu"
#include<omp.h>
using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){

  ///Classifier Parameters
  //下面这些参数通过程序开始运行时读入parameters.yml文件进行初始化
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];   //树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数
  structSize = (int)file["num_features"];  //每棵树的特征个数，也即每棵树的节点个数；树上每一个特征都作为一个决策节点
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}

void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs * structSize;
  //二维向量  包含全部尺度（scales）的扫描窗口，每个尺度包含totalFeatures个特征
  features = vector<vector<Feature> >(scales.size(), vector<Feature> (totalFeatures));
 
  //opencv中自带的一个随机数发生器的类RNG
  RNG& rng = theRNG();
  
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  //集合分类器基于n个基本分类器，每个分类器都是基于一个pixel comparisons（像素比较集）的；
  //pixel comparisons的产生方法：先用一个归一化的patch去离散化像素空间，产生所有可能的垂直和水平的pixel comparisons
  //然后我们把这些pixel comparisons随机分配给n个分类器，每个分类器得到完全不同的pixel comparisons（特征集合），
  //这样，所有分类器的特征组统一起来就可以覆盖整个patch了
  
  //用随机数去填充每一个尺度扫描窗口的特征
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0; s<scales.size(); s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
		  //第s种尺度的第i个特征  两个随机分配的像素点坐标
          features[s][i] = Feature(x1, y1, x2, y2);
      }
  }
  //Thresholds
  thrN = 0.5 * nstructs;

  //Initialize Posteriors  初始化后验概率
  //后验概率指每一个分类器对传入的图像片进行像素对比，每一个像素对比得到0或者1，所有的特征13个comparison对比，
  //连成一个13位的二进制代码x，然后索引到一个记录了后验概率的数组P(y|x)，y为0或者1（二分类），也就是出现x的
  //基础上，该图像片为y的概率是多少对n个基本分类器的后验概率做平均，大于0.5则判定其含有目标
  for (int i = 0; i<nstructs; i++) {
  //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons
  //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率
  //后验概率P(y|x)= #p/(#p+#n) ,#p和#n分别是正和负图像片的数目，也就是下面的pCounter和nCounter
  //初始化时，每个后验概率都得初始化为0；运行时候以下面方式更新：已知类别标签的样本（训练样本）通过n个分类器
  //进行分类，如果分类结果错误，那么响应的#p和#n就会更新，这样P(y|x)也相应更新了
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}

//该函数得到输入的image的用于树的节点，也就是特征组的特征（13位的二进制代码）
void FerNNClassifier::getFeatures(const cv::Mat& image, const int& scale_idx, vector<int>& fern){
  int leaf;  //叶子  树的最终节点
  //每一个每类器维护一个后验概率的分布，这个分布有2^d个条目（entries），这里d是像素比较pixel comparisons
  //的个数，这里是structSize，即13个comparison，所以会产生2^13即8,192个可能的code，每一个code对应一个后验概率
  int t;
  const int nCore = omp_get_num_procs();
#pragma omp parallel for num_threads(nCore)
  for (t= 0; t<nstructs; t++) {  //nstructs 表示树的个数 10
	  leaf = 0;
	  int f;
	  for ( f = 0; f<structSize; f++) {  //表示每棵树特征的个数 13
											//struct Feature 特征结构体有一个运算符重载 bool operator ()(const cv::Mat& patch) const
											//返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1
											//然后leaf就记录了这13位的二进制代码，作为特征
		  leaf = (leaf << 1) + features[scale_idx][t*nstructs + f](image);
	  }
	  fern[t] = leaf;
  }
 
}

float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;

  for (int i = 0; i < nstructs; i++) {
     // 后验概率posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      votes += posteriors[i][fern[i]];   //每棵树的每个特征值对应的后验概率累加值 作投票值？？
  }
  return votes;
}

//更新正负样本数，同时更新后验概率
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}

//训练集合分类器（n个基本分类器集合）
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  
  //thr_fern: 0.6 thrP定义为Positive thershold
  thrP = thr_fern * nstructs;                                    // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){    //为1表示正样本        //       if (Y[I] == 1) {
		      //measure_forest函数返回所有树的所有特征值对应的后验概率累加值
			  //该累加值如果小于正样本阈值，也就是是输入的是正样本，却被分类成负样本了
			  //出现分类错误，所以就把该样本添加到正样本库，同时更新后验概率
              if(measure_forest(ferns[i].first) <= thrP)      //         if (measure_forest(x) <= thrP)
			  ////更新正样本数，同时更新后验概率
                update(ferns[i].first, 1, 1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first, 0, 1);                 //             update(x,0,1);
          }
      }
  //}
}

//训练最近邻分类器
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf, dummy;
  vector<int> y(nn_examples.size(),0); //vector<T> v3(n, i); v3包含n个值为i的元素。y数组元素初始化为0
  y[0]=1;  //上面说到调用trainNN这个函数传入的nn_data样本集，只有一个pEx，在nn_data[0]
  vector<int> isin;

  for (int i=0; i<nn_examples.size(); i++){                          //  For each example
      //计算输入图像片与在线模型之间的相关相似度conf
      NNConf(nn_examples[i], isin, conf, dummy);                      //  Measure Relative similarity
	  //thr_nn: 0.65 阈值
	  //标签是正样本，如果相关相似度小于0.65 ，则认为其不含有前景目标，也就是分类错误了；这时候就把它加到正样本库
      if (y[i]==1 && conf <= thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                                 //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end

  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf)相关相似度, Conservative Similarity (csconf)保守相似度,
   * In pos. set|Id pos set|In neg. set (isin)
   */
void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  isin=vector<int>(3,-1);  //vector<T> v3(n, i); v3包含n个值为i的元素。 三个元素都是-1
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP, csmaxP, maxP=0;
  bool anyP=false;
  int maxPidx, validatedPart = ceil(pEx.size()*valid);  //ceil返回大于或者等于指定表达式的最小整数
  float nccN, maxN=0;
  bool anyN=false;
  //比较图像片p到在线模型M的距离（相似度），计算正样本最近邻相似度，也就是将输入的图像片与
  //在线模型中所有的图像片进行匹配，找出最相似的那个图像片，也就是相似度的最大值
  int i;
  const int nCore = omp_get_num_procs();
#pragma omp parallel for num_threads(nCore)
  for ( i=0; i<pEx.size(); i++){
      matchTemplate(pEx[i], example, ncc, CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5;  //计算匹配相似度
      if (nccP>ncc_thesame)  //ncc_thesame: 0.95
        anyP=true;
#pragma omp critical
	  {
		  if (nccP > maxP) {
			  maxP = nccP;    //记录最大的相似度以及对应的图像片index索引值
			  maxPidx = i;
			  if (i < validatedPart)
				  csmaxP = maxP;
		  }
	  }
  }
  //计算负样本最近邻相似度
  for (int i = 0; i < nEx.size(); i++) {
	  matchTemplate(nEx[i], example, ncc, CV_TM_CCORR_NORMED);     //measure NCC to negative examples
	  nccN = (((float*)ncc.data)[0] + 1)*0.5;
	  if (nccN > ncc_thesame)
		  anyN = true;
	 if (nccN > maxN)
			  maxN = nccN;
  }
  //set isin
  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  if (anyP) isin[0]=1;  
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  if (anyN) isin[2]=1; 
  
  //Measure Relative Similarity
  //相关相似度 = 正样本最近邻相似度 / （正样本最近邻相似度 + 负样本最近邻相似度）
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT, const vector<cv::Mat>& nExT){
  float fconf = 0;
  float tempFern = thr_fern;
  const int nCore = omp_get_num_procs();
#pragma omp parallel for num_threads(nCore)
  for (int i = 0; i < nXT.size(); i++) {
	  //所有基本分类器的后验概率的平均值如果大于thr_fern，则认为含有前景目标
	  //measure_forest返回的是所有后验概率的累加和，nstructs 为树的个数，也就是基本分类器的数目 
	  fconf = (float)measure_forest(nXT[i].first) / nstructs;
#pragma omp critical
	  {
		  if (fconf > tempFern)  //thr_fern: 0.6 thrP定义为Positive thershold
			  tempFern = fconf;  //取这个平均值作为 该集合分类器的 新的阈值，这就是训练
	  }
  }
  thr_fern = tempFern;
  vector <int> isin;
  float conf, dummy;

  for (int i=0; i<nExT.size(); i++){
      NNConf(nExT[i], isin, conf, dummy);
      if (conf > thr_nn)
        thr_nn = conf; //取这个最大相关相似度作为 该最近邻分类器的 新的阈值，这就是训练？？
  }
  
  if (thr_nn > thr_nn_valid)  //thr_nn_valid: 0.7
    thr_nn_valid = thr_nn;
}

//把正样本库（在线模型）包含的所有正样本显示在窗口上
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows, pEx[0].cols, CV_8U);
  double minval;
  Mat ex(pEx[0].rows, pEx[0].cols, pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    //minMaxLoc寻找矩阵（一维数组当作向量，用Mat定义）中最小值和最大值的位置. 
    minMaxLoc(pEx[i], &minval); //寻找pEx[i]的最小值
    pEx[i].copyTo(ex);
    ex = ex - minval;  //把像素亮度最小的像素重设为0，其他像素按此重设
	//Mat Mat::rowRange(int startrow, int endrow) const 为指定的行span创建一个新的矩阵头。
	//Mat Mat::rowRange(const Range& r) const   //Range 结构包含着起始和终止的索引值。
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows, (i+1)*pEx[i].rows));
    ex.convertTo(tmp, CV_8U);
  }
  imshow("Examples", examples);
}
