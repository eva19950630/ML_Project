#pragma once
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <random>

#define MaxNumClusters 255

struct pData {
	double X;
	double Y;
	int ClassKind;
};

namespace ML_Project1 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Drawing::Imaging;
	using namespace System::Runtime::InteropServices;
	using namespace std;

	/// <summary>
	/// Form1 的摘要
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Bitmap^ myBitmap;
		Graphics^ g;
		Brush^ bshDraw;
		Pen^ penDraw;
		String^ Filename1;
		unsigned char PointSize, PointSize1, PointSize2, Distribution;
		pData *InputData;
		double Pi, CenterX, CenterY;
		int ClassKind, MethodCodeValue, NumberOfData, NumberOfPoint, MaxSizeOfData;
		int imW, imH, X_Cur, Y_Cur, RangeX, RangeY, NumOfCluster, NumOfClass, NumClass1, NumClass2;
		bool HandFlag;
		//Bayesian Parameters
		double MeanX1, MeanY1, Sigma2X1, Sigma2Y1, SigmaX1, SigmaY1, SigmaXY1, detA1, Correlation1, Correlation12, PClass1, *PxyClass1;
		double MeanX2, MeanY2, Sigma2X2, Sigma2Y2, SigmaX2, SigmaY2, SigmaXY2, detA2, Correlation2, Correlation22, PClass2, *PxyClass2;
		//Regression--Linear & LinearLn	
		double LR_a1, LR_a0;
		//Regression--Nonlinear
		double **A, *B, *NLcoef; //AX=B ==> solve equation a0+a1X+...+adX^dfor (a0,a1,...,ad)=NLcoef[]
		int NLdegree; //polynomial degree
		//clustering, K-means, FCM, EM used
		unsigned char NumOfClusters;
		int *BackupClassKind;
		bool STOPFlag;
		unsigned char *InputDataClusterType;
		pData *ClusterCenter;
		unsigned short *Radius;
		double **dist;
		double **U;
		//K-NN Classification
		unsigned char kNNs;
		int totalCTestData, MaxKNN;
		short **ALLNNs, *ALLCountClass1, *ALLCountClass2;
		//K-NN Regression
		double **BDdist; //distance between test and input Data
		int totalRTestData, **NNs;
		bool BuiltkNNFlag;
		//Perceptron, LVQ
		pData *W;
		double Bias;

	public:
		Brush^ ClassToColor(int c) { //Brush->Color depend on Classkind
			Brush^ tmpBrush;
			switch (c) {
			case -2: //black
				tmpBrush = gcnew SolidBrush(Color::Black);
				break;
			case -1: //blue
				tmpBrush = gcnew SolidBrush(Color::Blue);
				break;
			case 0: //Gray
				tmpBrush = gcnew SolidBrush(Color::Gray);
				break;
			case 1: //red
				tmpBrush = gcnew SolidBrush(Color::Red);
				break;
			case 2: //Green
				tmpBrush = gcnew SolidBrush(Color::Green);
				break;
			case 3: //Cyan
				tmpBrush = gcnew SolidBrush(Color::Cyan);
				break;
			case 4: //Magenta
				tmpBrush = gcnew SolidBrush(Color::Magenta);
				break;
			case 5: //Yellow
				tmpBrush = gcnew SolidBrush(Color::Yellow);
				break;
			case 6: //Brown
				tmpBrush = gcnew SolidBrush(Color::Brown);
				break;
			case 7: //Purple
				tmpBrush = gcnew SolidBrush(Color::Purple);
				break;
			default: //Green
				tmpBrush = gcnew SolidBrush(Color::Green);
				break;
			}//switch
			return tmpBrush;
		}//ClassToColor
		Pen^ ClassToPenColor(int c) { //Brush->Color depend on Classkind
			Pen^ tmpPen;
			switch (c) {
			case -2: //black
				tmpPen = gcnew Pen(Color::Black);
				break;
			case -1: //blue
				tmpPen = gcnew Pen(Color::Blue);
				break;
			case 0: //Gray
				tmpPen = gcnew Pen(Color::Gray);
				break;
			case 1: //red
				tmpPen = gcnew Pen(Color::Red);
				break;
			case 2: //Green
				tmpPen = gcnew Pen(Color::Green);
				break;
			case 3: //Cyan
				tmpPen = gcnew Pen(Color::Cyan);
				break;
			case 4: //Magenta
				tmpPen = gcnew Pen(Color::Magenta);
				break;
			case 5: //Yellow
				tmpPen = gcnew Pen(Color::Yellow);
				break;
			case 6: //Brown
				tmpPen = gcnew Pen(Color::Brown);
				break;
			case 7: //Purple
				tmpPen = gcnew Pen(Color::Purple);
				break;
			default: //Green
				tmpPen = gcnew Pen(Color::Green);
				break;
			}//switch
			return tmpPen;
		}//ClassToColor
		void NewPublicVariables(int MaxNumberOfData) {
			InputData = new pData[MaxNumberOfData];
			PxyClass1 = new double[MaxNumberOfData];
			PxyClass2 = new double[MaxNumberOfData];
			//Clustering, K-means, FCM, EM used
			ClusterCenter= new pData[MaxNumClusters]; //ClusterCenter
			InputDataClusterType= new unsigned char[MaxNumberOfData];
			BackupClassKind=new int[MaxNumberOfData];
			Radius = new unsigned short[MaxNumClusters];
			dist= new double*[MaxNumberOfData]; //dist[][]-->distance between data and data
			for (int k=0; k<MaxNumberOfData; k++)
				dist[k]= new double[MaxNumberOfData];
			U = new double*[MaxNumberOfData]; //U[][]-->FCM Uij
			for (int k = 0; k<MaxNumberOfData; k++)
				U[k] = new double[MaxNumberOfData];
			//K-NN Classification
			ALLNNs = new short*[totalCTestData]; //ALLNNs
			for (int k = 0; k<totalCTestData; k++)
				ALLNNs[k] = new short[255];
			ALLCountClass1 = new short[totalCTestData];
			ALLCountClass2 = new short[totalCTestData];
			//K-NN Regression
			BDdist = new double*[totalRTestData]; //BDdist
			for (int k = 0; k<totalRTestData; k++)
				BDdist[k] = new double[255];
			NNs = new int*[totalRTestData]; //NNs
			for (int k = 0; k<totalRTestData; k++)
				NNs[k] = new int[255];
			//Perceptron, LVQ
			W = new pData[MaxNumberOfData]; //Neural Networks(Perceptron, BP, LVQ)
		}
		void DeletePublicVariables(unsigned short MaxNumberOfData) {
			delete [] InputData;
			delete [] PxyClass1;
			delete [] PxyClass2;
			//Clustering, K-means, FCM, EM used
			delete [] ClusterCenter;
			delete [] InputDataClusterType;
			delete [] BackupClassKind;
			delete [] Radius;
			for (int k=0; k<MaxNumberOfData; k++)
				delete [] dist[k];
			delete [] dist;
			for (int k = 0; k<MaxNumberOfData; k++)
				delete[] U[k];
			delete[] U;
			//K-NN Classification
			for (int k = 0; k<totalCTestData; k++)
				delete[] ALLNNs[k];
			delete[] ALLNNs;
			delete [] ALLCountClass1;
			delete [] ALLCountClass2;
			//K-NN Regression
			for (int k = 0; k<totalRTestData; k++)
				delete[] BDdist[k];
			for (int k = 0; k<totalRTestData; k++)
				delete[] NNs[k];
			// Perceptron, LVQ
			delete[] W; //Neural Networks(Perceptron, BP, LVQ)
		}
		double Sgn(double Num1) {
			return (Num1 >= 0.0) ? 1.0 : -1.0;
		}
		void CalculateMeanSigma2() {
			MeanX1 = 0.0;
			MeanY1 = 0.0;
			MeanX2 = 0.0;
			MeanY2 = 0.0;
			NumClass1 = 0;
			NumClass2 = 0;
			for (int i = 0; i < NumberOfData; i++) {
				if (InputData[i].ClassKind == 1) {
					MeanX1 += InputData[i].X;
					MeanY1 += InputData[i].Y;
					NumClass1++;
				}
				else {
					MeanX2 += InputData[i].X;
					MeanY2 += InputData[i].Y;
					NumClass2++;
				}
			}
			MeanX1 = MeanX1 / NumClass1;
			MeanY1 = MeanY1 / NumClass1;
			MeanX2 = MeanX2 / NumClass2;
			MeanY2 = MeanY2 / NumClass2;

			Sigma2X1 = 0.0;
			Sigma2Y1 = 0.0;
			Sigma2X2 = 0.0;
			Sigma2Y2 = 0.0;
			SigmaXY1 = 0.0;
			SigmaXY2 = 0.0;
			for (int i = 0; i < NumberOfData; i++) {
				if (InputData[i].ClassKind == 1) {
					Sigma2X1 += pow((InputData[i].X - MeanX1), 2);
					Sigma2Y1 += pow((InputData[i].Y - MeanY1), 2);
					SigmaXY1 += (InputData[i].X - MeanX1)*(InputData[i].Y - MeanY1);
				}
				else {
					Sigma2X2 += pow((InputData[i].X - MeanX2), 2);
					Sigma2Y2 += pow((InputData[i].Y - MeanY2), 2);
					SigmaXY2 += (InputData[i].X - MeanX2)*(InputData[i].Y - MeanY2);
				}
			}
			if (NumClass1 > 0) {
				if (NumClass1 == 1 || !checkBox_Unbiased->Checked) {
					Sigma2X1 /= NumClass1;
					Sigma2Y1 /= NumClass1;
					SigmaXY1 /= NumClass1;
				}
				else {
					Sigma2X1 /= (NumClass1 - 1);
					Sigma2Y1 /= (NumClass1 - 1);
					SigmaXY1 /= (NumClass1 - 1);
				}
			}
			if (NumClass2 > 0) {
				if (NumClass2 == 1 || !checkBox_Unbiased->Checked) {
					Sigma2X2 /= NumClass2;
					Sigma2Y2 /= NumClass2;
					SigmaXY2 /= NumClass2;
				}
				else {
					Sigma2X2 /= (NumClass2 - 1);
					Sigma2Y2 /= (NumClass2 - 1);
					SigmaXY2 /= (NumClass2 - 1);
				}
			}
			SigmaX1 = sqrt(Sigma2X1);
			SigmaY1 = sqrt(Sigma2Y1);
			SigmaX2 = sqrt(Sigma2X2);
			SigmaY2 = sqrt(Sigma2Y2);
			Correlation1 = SigmaXY1 / (SigmaX1*SigmaY1);
			Correlation12 = Correlation1*Correlation1;
			Correlation2 = SigmaXY2 / (SigmaX2*SigmaY2);
			Correlation22 = Correlation2*Correlation2;
			detA1 = Sigma2X1*Sigma2Y1 - SigmaXY1*SigmaXY1;
			detA2 = Sigma2X2*Sigma2Y2 - SigmaXY2*SigmaXY2;
		}
		double evalPxy1(pData Data1) {
			double dx, dy, dx2, dy2, c1, Ndist, tmp;
			//double Pi=4.0*atan(1.0);
			//Class1 (Red)
			dx = Data1.X - MeanX1;
			dy = Data1.Y - MeanY1;
			dx2 = dx*dx;
			dy2 = dy*dy;
			if (Correlation12 != 1.0)
				c1 = 1.0 / (1.0 - Correlation12);
			else
				c1 = 1.0;

			if (detA1 == 0.0) {
				Ndist = dx2 - 2.0*dx*dy + dy2;
				tmp = exp(-0.5*Ndist);
			}
			else {
				Ndist = dx2 / Sigma2X1 - 2.0*Correlation12*dx*dy / SigmaXY1 + dy2 / Sigma2Y1;
				tmp = exp(-0.5*c1*Ndist) / (2.0*Pi*sqrt(detA1));
			}
			
			return tmp;
		}
		double evalPxy2(pData Data1) {
			double dx, dy, dx2, dy2, c1, Ndist, tmp;
			//double Pi=4.0*atan(1.0);
			//Class1 (Red)
			dx = Data1.X - MeanX2;
			dy = Data1.Y - MeanY2;
			dx2 = dx*dx;
			dy2 = dy*dy;
			if (Correlation22 != 1.0)
				c1 = 1.0 / (1.0 - Correlation22);
			else
				c1 = 1.0;

			if (detA2 == 0.0) {
				Ndist = dx2 - 2.0*dx*dy + dy2;
				tmp = exp(-0.5*Ndist);
			}
			else {
				Ndist = dx2 / Sigma2X2 - 2.0*Correlation22*dx*dy / SigmaXY2 + dy2 / Sigma2Y2;
				tmp = exp(-0.5*c1*Ndist) / (2.0*Pi*sqrt(detA2));
			}
			
			return tmp;
		}
		void CalculateBayesianProb() {
			double dx, dy, dx2, dy2, c1, Ndist;
			//double Pi=4.0*atan(1.0);

			PClass1 = (double)NumClass1 / NumberOfData;
			PClass2 = (double) 1.0 - PClass1;

			for (int i = 0; i < NumberOfData; i++) {
				//Class1 (Red)
				dx = InputData[i].X - MeanX1;
				dx2 = dx*dx;
				dy2 = dy*dy;
				if (Correlation12 != 1.0)
					c1 = 1.0 / (1.0 - Correlation12);
				else
					c1 = 1.0;

				if (detA1 == 0.0) {
					Ndist = dx2 - 2.0*dx*dy + dy2;
					PxyClass1[i] = exp(-0.5*Ndist);
				}
				else {
					Ndist = dx2 / Sigma2X1 - 2.0*Correlation12*dx*dy / SigmaXY1 + dy2 / Sigma2Y1;
					PxyClass1[i] = exp(-0.5*c1*Ndist) / (2.0*Pi*sqrt(detA1));
				}

				//Class2 (Blue)
				dx = InputData[i].X - MeanX2;
				dy = InputData[i].Y - MeanY2;
				dx2 = dx*dx;
				dy2 = dy*dy;
				if (Correlation22 != 1.0)
					c1 = 1.0 / (1.0 - Correlation22);
				else
					c1 = 1.0;

				if (detA2 == 0.0) {
					Ndist = dx2 - 2.0*dx*dy + dy2;
					PxyClass2[i] = exp(-0.5*Ndist);
				}
				else {
					Ndist = dx2 / Sigma2X2 - 2.0*Correlation22*dx*dy / SigmaXY2 + dy2 / Sigma2Y2;
					PxyClass2[i] = exp(-0.5*c1*Ndist) / (2.0*Pi*sqrt(detA2));
				}
			}
		}
		void BayesMAP() {
			CalculateMeanSigma2();
			CalculateBayesianProb();
		}
		void GaussEliminationPivot(int n){
			double tmp, pvt;
			int index_pvt, *pivot;

			pivot = new int[n];

			for (int j = 0; j < n-1; j++) {
				pvt = abs(A[j][j]);
				pivot[j] = j;
				index_pvt = j;
				//find pivot
				for (int i = j+1; i < n; i++) {
					if (abs(A[i][j]) > pvt) {
						pvt = abs(A[i][j]);
						index_pvt = i;
					}//if
				}//for i

				//switch row pivot[j] and row index_pvt
				if (pivot[j] != index_pvt) {
					for (int i = 0; i < n; i++) {
						tmp = A[pivot[j]][i];
						A[pivot[j]][i] = A[index_pvt][i];
						A[index_pvt][i] = tmp;
					}//for i
					tmp = B[pivot[j]];
					B[pivot[j]] = B[index_pvt];
					B[index_pvt] = tmp;
				}//if

				for (int i = j+1; i < n; i++)
					A[i][j] /= A[j][j];

				//produce Upper triangle matrix
				for (int i = j+1; i < n; i++) {
					for (int k = j+1; k < n; k++) {
						A[i][k] -= A[i][j]*A[j][k];
					}//for k
					B[i] -= A[i][j] * B[j];
				}//for i
			}//for j

			//back substitution
			//for (int i = 0; i < n; i++)
			//	NLcoef[i] =0.0;
			NLcoef[n-1] = B[n-1] / A[n-1][n-1];
			for (int j = n-2; j >= 0; j--) {
				NLcoef[j] = B[j];
				for (int k = n-1; k > j; k--) {
					NLcoef[j] -= NLcoef[k]*A[j][k];
				}//for k
				NLcoef[j] /= A[j][j];
			}//for j
			delete [] pivot;
		} //解聯立方程式之高斯消去法
		void LinearRegression(){	
			double sumX = 0;
			double sumY = 0;
			double sumXY = 0;
			double XX = 0;
			double sumXX = 0;
			for (int i = 0; i < NumberOfData; i++) {
				sumX += InputData[i].X;
				sumY += InputData[i].Y;
				sumXY += InputData[i].X * InputData[i].Y;
				XX = InputData[i].X * InputData[i].X;
				sumXX += XX;
			}
			LR_a1 = ((NumberOfData*sumXY) - (sumX*sumY)) / ((NumberOfData*sumXX) - (sumX*sumX));
			LR_a0 = (sumY/NumberOfData) - LR_a1 * (sumX/NumberOfData);
		} //Linear --線性迴歸公用程式
		void LinearRegressionLn(){
			double sumX = 0;
			double sumY = 0;
			double sumXY = 0;
			double XX = 0;
			double sumXX = 0;
			double shiftX = 0;
			double shiftY = 0;
			double tmpY = 0;
			double a0 = 0;
			for (int i = 0; i < NumberOfData; i++) {		
				shiftX = InputData[i].X+2.0;
				shiftY = InputData[i].Y+2.0;
				tmpY = log(shiftY);
				sumX += shiftX;			
				sumY += tmpY;
				sumXY += shiftX * tmpY;
				XX = shiftX * shiftX;
				sumXX += XX;
			}
			LR_a1 = ((NumberOfData*sumXY) - (sumX*sumY)) / ((NumberOfData*sumXX) - (sumX*sumX));
			a0 = (sumY/NumberOfData) - LR_a1 * (sumX/NumberOfData);
			LR_a0 = exp(a0);
		} //Linear—log e 線性迴歸作非線性迴歸公用程式
		void NonlinearRegression(int degree){
			for (int i = 0; i < degree+1; i++) {
				for (int j = 0; j < degree+1; j++) {
					for (int k = 0; k < NumberOfData; k++) {
						A[i][j] += pow(InputData[k].X, i+j);
					}
				}
			}
			for (int j = 0; j < degree+1; j++) {
				for (int k = 0; k < NumberOfData; k++) {
					B[j] += pow(InputData[k].X, j) * InputData[k].Y;
				}
			}
			GaussEliminationPivot(degree+1);
		} //Non-linear—非線性迴歸公用程式
		double rand01() {
			return rand() / double(RAND_MAX);
		} //產生介於0.0~1.0間的亂數
		int rand_m(int Num1) {
			return rand() % Num1;
		} //產生介於0~Num1-1間的整數亂數
		double MAX(double Num1, double Num2){
			if (Num2 > Num1) {
				return Num2;
			} else {
				return Num1;
			}
		}//取Num1和Num2兩者較大的數
		double MIN(double Num1, double Num2){
			if (Num2 < Num1) {
				return Num2;
			} else {
				return Num1;
			}
		}//取Num1和Num2兩者較小的數
		void K_Means(unsigned char K_Clusters){
			int count = 0;
			double sumX = 0.0;
			double sumY = 0.0;
			double d = 0.0;
			double tmpDelta = 0.0;
			double maxDelta = 0.0;
			switch (comboBox_Kmeans_Option->SelectedIndex) {
				case 0:
					//隨機取初始中心點
					for (int i = 0; i < K_Clusters; i++) {
						ClusterCenter[i] = InputData[rand_m(NumberOfData)];
						ClusterCenter[i].ClassKind = i;
					}	
					while (maxDelta >= d) {
						d = Convert::ToDouble(textBox_delta->Text);
						maxDelta = 0.0;
						//找每個點到初始中心點距離
						for (int i = 0; i < NumberOfData; i++) {
							double X1 = InputData[i].X;
							double Y1 = InputData[i].Y;
							for (int k = 0; k < K_Clusters; k++) {
								double X2 = ClusterCenter[k].X;
								double Y2 = ClusterCenter[k].Y;
								dist[i][k] = pow(X1 - X2, 2) + pow(Y1 - Y2, 2);
							}
						}
						//找每個點離哪個初始中心點距離最短
						for (int i = 0; i < NumberOfData; i++) {
							InputData[i].ClassKind = 0;
							for (int k = 1; k < K_Clusters; k++) {
								if (dist[i][k] < dist[i][0]) {
									dist[i][0] = dist[i][k];
									InputData[i].ClassKind = k;
								}
							}
						}
						//找分群後之真正中心點 算術平均 分別算群內x點總和/群內總點數 y點總和/群內總點數 得到中心
						for (int k = 0; k < K_Clusters; k++) {
							sumX = 0.0;
							sumY = 0.0;
							count = 0;	
							double X3 = ClusterCenter[k].X;
							double Y3 = ClusterCenter[k].Y;
							for (int i = 0; i < NumberOfData; i++) {
								if (InputData[i].ClassKind == k) {
									sumX += InputData[i].X;
									sumY += InputData[i].Y;
									count++;
								}			
								ClusterCenter[k].X = sumX / count;
								ClusterCenter[k].Y = sumY / count;
								ClusterCenter[k].ClassKind = k;	
								tmpDelta = sqrt(pow((sumX / count) - X3, 2) + pow((sumY / count) - Y3, 2));
							}		
							if (tmpDelta > maxDelta) {
								maxDelta = tmpDelta;
							}
							//maxDelta = MAX(maxDelta, tmpDelta);
						}				
					}
					break;
			}
		} //k_means公用程式
		void FCM(unsigned char K_Clusters){
			double b = 2.0;
			double sumDist = 0.0;
			double sumUij = 0.0;
			double sumUijX = 0.0;
			double sumUijY = 0.0;
			double d = 0.0;
			double tmpDelta = 0.0;
			double maxDelta = 0.0;
			switch (comboBox_Kmeans_Option->SelectedIndex) {
				case 0:
					//隨機取初始中心點
					for (int i = 0; i < K_Clusters; i++) {
						ClusterCenter[i] = InputData[rand_m(NumberOfData)];
						ClusterCenter[i].ClassKind = i;
					}
					while (maxDelta >= d) {
						d = Convert::ToDouble(textBox_delta->Text);
						maxDelta = 0.0;
						//找每個點到初始中心點距離 & 算每個點到各初始中心點距離加總 & Uij
						for (int i = 0; i < NumberOfData; i++) {
							double X1 = InputData[i].X;
							double Y1 = InputData[i].Y;	
							sumDist = 0.0;
							for (int k = 0; k < K_Clusters; k++) {
								double X2 = ClusterCenter[k].X;
								double Y2 = ClusterCenter[k].Y;
								dist[i][k] = sqrt(pow(X1 - X2, 2) + pow(Y1 - Y2, 2));
								if (dist[i][k] == 0.0) {
									dist[i][k] = pow(10, 38);
								}	
								sumDist += 1.0 / dist[i][k];
							}
							for (int k = 0; k < K_Clusters; k++) {
								U[i][k] = (1.0 / dist[i][k]) / sumDist;
							}
						}
						//找新中心點
						for (int k = 0; k < K_Clusters; k++) {
							double X3 = ClusterCenter[k].X;
							double Y3 = ClusterCenter[k].Y;
							sumUij = 0.0;
							sumUijX = 0.0;
							sumUijY = 0.0;
							for (int i = 0; i < NumberOfData; i++) {
								sumUij += pow(U[i][k], b);
								if (sumUij == 0.0) {
									sumUij = pow(10, 38);
								}
								sumUijX += (pow(U[i][k], b) * InputData[i].X);
								sumUijY += (pow(U[i][k], b) * InputData[i].Y);
							}
							ClusterCenter[k].X = sumUijX / sumUij;
							ClusterCenter[k].Y = sumUijY / sumUij;
							tmpDelta = sqrt(pow(ClusterCenter[k].X - X3, 2) + pow(ClusterCenter[k].Y - Y3, 2));
							if (tmpDelta > maxDelta) {
								maxDelta = tmpDelta;
							}
						}
					}				
					//找每個點到新中心點距離
					for (int i = 0; i < NumberOfData; i++) {
						double X1 = InputData[i].X;
						double Y1 = InputData[i].Y;
						for (int k = 0; k < K_Clusters; k++) {
							double X2 = ClusterCenter[k].X;
							double Y2 = ClusterCenter[k].Y;
							dist[i][k] = pow(X1 - X2, 2) + pow(Y1 - Y2, 2);
						}
					}
					//找每個點隸屬哪一群
					for (int i = 0; i < NumberOfData; i++) {
						InputData[i].ClassKind = 0;
						for (int k = 1; k < K_Clusters; k++) {
							if (dist[i][k] < dist[i][0]) {
								dist[i][0] = dist[i][k];
								InputData[i].ClassKind = k;
							}
						}
					}
					break;
			}
		} //Fuzzy C-Means公用程式
		short FindMaxKNN(){
			//MaxKNN = NumberOfData - 1;
			return NumberOfData - 1;
		} //計算每個資料檔k-NN的上限，即k最大值會依資料數量而異。
		void Create_kNN_Contour_Table() {
			//double tmp = 0.0;
			double Min = 0.0;
			double tmpMin = 0.0;
			int index = 0;
			double *D;
			double shiftX = 0.0;
			double shiftY = 0.0;
			D = new double[NumberOfData];
			for (int i = 0; i < totalCTestData; i++) {
				ALLCountClass1[i] = 0;
				ALLCountClass2[i] = 0;
			}
			for (int x = 0; x < imW; x++) {
				for (int y = 0; y < imH; y++) {
					shiftX = (double)(x - CenterX) / CenterX;
					shiftY = (double)(CenterY - y) / CenterY;
					//找每個測試點到各資料點的距離
					for (int j = 0; j < NumberOfData; j++) {
						D[j] = sqrt(pow(shiftX - InputData[j].X, 2) + pow(shiftY - InputData[j].Y, 2));
					}
					//找每個測試點的k個最近鄰居 & 將找到的最近鄰居分兩類 算出個別數量
					for (int k = 0; k < kNNs; k++) {
						index = 0;
						Min = D[0];
						for (int i = 0; i < NumberOfData; i++) {
							tmpMin = D[i];
							if (tmpMin <= Min) {
								Min = tmpMin;
								index = i;
							}
						}
						ALLNNs[y*imW + x][k] = index;
						D[index] = pow(10, 38);

						if (InputData[index].ClassKind == 1) {
							ALLCountClass1[y*imW + x]++;
						}
						else if (InputData[index].ClassKind == -1) {
							ALLCountClass2[y*imW + x]++;
						}
					}
				}
			}
			delete[] D;
		} //showContour時所需重建的每個資料點之k-NN Table。
		void BuildAllkNNRegTable() {
			double shiftX = 0.0;
			int index = 0;
			double Min = 0.0;
			double tmpMin = 0.0;
			for (int x = 0; x < imW; x++) {
				shiftX = (double)(x - CenterX) / CenterX;
				//找每個測試點到各資料點的距離
				for (int j = 0; j < NumberOfData; j++) {
					BDdist[x][j] = abs(shiftX - InputData[j].X);
				}
				for (int k = 0; k < kNNs; k++) {
					index = 0;
					Min = BDdist[x][0];
					for (int t = 0; t < NumberOfData; t++) {
						tmpMin = BDdist[x][t];
						if (tmpMin <= Min) {
							Min = tmpMin;
							index = t;
						}
					}
					NNs[x][k] = index;
					BDdist[x][index] = pow(10, 38);
				}
			}
		} //showRegression時所需重建的每個資料點之k-NN Table。
		double PerceptronClassify(pData Sample){
			double a = 0.0;
			a = Sgn((W[0].X * Sample.X) + (W[0].Y * Sample.Y) + Bias);
			return a;
		} //計算每個測試資料屬於哪一類?
		void Perceptron_CTrain(){
			W[0].X = rand01() * 2 - 1;
			W[0].Y = rand01() * 2 - 1;
			double a = 0.0;
			int e = 0;
			int sumE = 0;
			int count = 0;
			int Max = Convert::ToInt32(textBox_MaxIter->Text);
			double alpha = Convert::ToDouble(textBox_ini->Text);
			bool StopWhile = false;
			while (!StopWhile) {
				for (int i = 0; i < NumberOfData; i++) {
					a = PerceptronClassify(InputData[i]);
					e = InputData[i].ClassKind - a;
					sumE += abs(e);
					W[0].X += e * InputData[i].X * alpha;
					W[0].Y += e * InputData[i].Y * alpha;
					Bias += e * alpha;
				}
				count++;
				if (sumE == 0 || count > Max) {
					break;
				}
			}	
		} //Training weights for Perceptron Classification。
		double PerceptronRegression(double testX, int T_Function){
			double n = 0.0;
			n = W[0].X * testX + Bias;
			switch (T_Function) {
				case 1:
					return n;
					break;
				case 3:
					return 2.0 / (1 + exp(-2 * n)) - 1;
					break;
				default:
					break;
			}
		} //計算每個測試資料迴歸值?
		void Perceptron_RTrain(int T_Function){
			W[0].X = rand01() * 2 - 1;
			double py = 0.0;
			double dy = 0.0;
			double fy = 0.0;
			double E = 0.0;
			double alpha = Convert::ToDouble(textBox_ini->Text);
			double epsilon = Convert::ToDouble(textBox_eps->Text);
			int Max = Convert::ToInt32(textBox_MaxIter->Text);
			bool StopWhile = false;
			int count = 0;
			while (!StopWhile) {
				E = 0.0;
				switch (T_Function){
					case 1:
						for (int i = 0; i < NumberOfData; i++) {
							py = PerceptronRegression(InputData[i].X, T_Function);
							dy = InputData[i].Y - py;
							W[0].X += alpha * dy * InputData[i].X;
							E += pow(dy, 2) * 0.5;
							Bias += alpha * dy;
						}
						count++;
						break;
					case 3:
						for (int i = 0; i < NumberOfData; i++) {
							py = PerceptronRegression(InputData[i].X, T_Function);
							dy = InputData[i].Y - py;
							fy = 1.0 - py * py;
							W[0].X += dy * fy * InputData[i].X;
							E += pow(dy, 2) * 0.5;
							Bias += alpha * dy;
						}
						count++;
						break;
					default:
						break;
				}
				if (E < epsilon || count > Max) {
					break;
				}
			}
		} // Training weights for Perceptron Regression

	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: 在此加入建構函式程式碼
			//
		}

	private: System::Windows::Forms::ToolStripMenuItem^  newToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  openToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveAsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  exitToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveToolStripMenuItem;
	private: System::Windows::Forms::SaveFileDialog^  saveFileDialog1;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
	private: System::Windows::Forms::ComboBox^  comboBox_CS;
	private: System::Windows::Forms::ToolStripMenuItem^  saveImageAsToolStripMenuItem;
	private: System::Windows::Forms::GroupBox^  groupBox8;
	private: System::Windows::Forms::ComboBox^  comboBox_Run;
	private: System::Windows::Forms::GroupBox^  groupBox9;
	private: System::Windows::Forms::ComboBox^  comboBox_classify;
	private: System::Windows::Forms::GroupBox^  groupBox10;
	private: System::Windows::Forms::Label^  label14;
	private: System::Windows::Forms::ComboBox^  comboBox_clusters;
	private: System::Windows::Forms::ComboBox^  comboBox_clustering;
	private: System::Windows::Forms::GroupBox^  groupBox11;
	private: System::Windows::Forms::Label^  label15;
	private: System::Windows::Forms::ComboBox^  comboBox_NL_degree;
	private: System::Windows::Forms::ComboBox^  comboBox_regression;
	private: System::Windows::Forms::GroupBox^  groupBox12;
	private: System::Windows::Forms::GroupBox^  groupBox13;
	private: System::Windows::Forms::TextBox^  textBox_MaxIter;
	private: System::Windows::Forms::TextBox^  textBox_delta;
	private: System::Windows::Forms::Label^  label17;
	private: System::Windows::Forms::Label^  label16;
	private: System::Windows::Forms::GroupBox^  groupBox14;
	private: System::Windows::Forms::TextBox^  textBox3;
	private: System::Windows::Forms::Label^  label18;
	private: System::Windows::Forms::CheckBox^  checkBox_Unbiased;
	private: System::DirectoryServices::DirectoryEntry^  directoryEntry1;
	private: System::Windows::Forms::ToolStripMenuItem^  showDataToolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^  clearImageToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showResultToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showMeansToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showContourToolStripMenuItem;
	private: System::Windows::Forms::MenuStrip^  menuStrip1; 
	private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  imagingToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  dataEditToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  aboutToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  applicationsToolStripMenuItem;
	private: System::Windows::Forms::ToolStrip^  toolStrip1;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton1;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton2;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton3;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton4;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton5;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton6;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton7;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton8;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton9;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton10;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton11;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton12;
	private: System::Windows::Forms::ToolStripButton^  toolStripButton13;
	private: System::Windows::Forms::CheckBox^  checkBox1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TextBox^  textBox_X;
	private: System::Windows::Forms::TextBox^  textBox_Y;
	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::ComboBox^  comboBox_psize;
	private: System::Windows::Forms::GroupBox^  groupBox2;
	private: System::Windows::Forms::TextBox^  textBox_datasize;
	private: System::Windows::Forms::GroupBox^  groupBox3;
	private: System::Windows::Forms::TextBox^  textBox_MaxSize;
	private: System::Windows::Forms::Button^  button_Clear_Click;
	private: System::Windows::Forms::Button^  button_Run;
	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::GroupBox^  groupBox4;
	private: System::Windows::Forms::RadioButton^  radioButton_Group;
	private: System::Windows::Forms::RadioButton^  radioButton_Single;
	private: System::Windows::Forms::GroupBox^  groupBox5;
	private: System::Windows::Forms::RadioButton^  radioButton_NC;
	private: System::Windows::Forms::RadioButton^  radioButton_C2;
	private: System::Windows::Forms::RadioButton^  radioButton_CS;
	private: System::Windows::Forms::RadioButton^  radioButton_C1;
	private: System::Windows::Forms::RadioButton^  radioButton10;
	private: System::Windows::Forms::GroupBox^  groupBox6;
	private: System::Windows::Forms::GroupBox^  groupBox7;
	private: System::Windows::Forms::HScrollBar^  hScrollBar1;
	private: System::Windows::Forms::TextBox^  textBox8;
	private: System::Windows::Forms::TextBox^  textBox7;
	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::TextBox^  textBox6;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::TextBox^  textBox5;
	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::RadioButton^  radioButton7;
	private: System::Windows::Forms::TextBox^  textBox_Filename;
	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::RichTextBox^  richTextBox1;
	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::Label^  label9;
	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::Label^  label11;
	private: System::Windows::Forms::Label^  label12;
	private: System::Windows::Forms::Label^  label13;
	private: System::Windows::Forms::ToolStripMenuItem^  showRegressionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showClusteredToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  showClusterCenterToolStripMenuItem1;
	private: System::Windows::Forms::GroupBox^  groupBox15;
	private: System::Windows::Forms::CheckBox^  checkBox_ShowRange;
	private: System::Windows::Forms::GroupBox^  groupBox16;
	private: System::Windows::Forms::ComboBox^  comboBox_Kmeans_Option;
	private: System::Windows::Forms::Label^  label19;
	private: System::Windows::Forms::GroupBox^  groupBox17;
	private: System::Windows::Forms::ComboBox^  comboBox_Weight;
	private: System::Windows::Forms::Label^  label22;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::Label^  label21;
	private: System::Windows::Forms::ComboBox^  comboBox_kNN;
	private: System::Windows::Forms::Label^  label20;
	private: System::Windows::Forms::CheckBox^  checkBox2;
	private: System::Windows::Forms::GroupBox^  groupBox18;
	private: System::Windows::Forms::GroupBox^  groupBox19;
	private: System::Windows::Forms::ComboBox^  comboBox1;
	private: System::Windows::Forms::GroupBox^  groupBox20;
	private: System::Windows::Forms::GroupBox^  groupBox21;
	private: System::Windows::Forms::ComboBox^  comboBox5;
	private: System::Windows::Forms::TextBox^  textBox11;
	private: System::Windows::Forms::Label^  label27;
	private: System::Windows::Forms::ComboBox^  comboBox_P_Function;
	private: System::Windows::Forms::TextBox^  textBox10;
	private: System::Windows::Forms::Label^  label26;
	private: System::Windows::Forms::Label^  label25;
	private: System::Windows::Forms::TextBox^  textBox9;
	private: System::Windows::Forms::Label^  label24;
	private: System::Windows::Forms::TextBox^  textBox_ini;
	private: System::Windows::Forms::Label^  label23;
	private: System::ComponentModel::IContainer^  components;
	private: System::Windows::Forms::TextBox^  textBox_eps;
	private: System::Windows::Forms::Label^  label28;
	private: System::Windows::Forms::ImageList^  imageList1;

	private:
		/// <summary>
		/// 設計工具所需的變數。
		/// </summary>


	protected:
		/// <summary>
		/// 清除任何使用中的資源。
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器
		/// 修改這個方法的內容。
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			System::ComponentModel::ComponentResourceManager^  resources = (gcnew System::ComponentModel::ComponentResourceManager(Form1::typeid));
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->newToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveAsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveImageAsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->imagingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showDataToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->clearImageToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showResultToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showMeansToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showContourToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showRegressionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showClusteredToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->showClusterCenterToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->dataEditToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->aboutToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->applicationsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->toolStrip1 = (gcnew System::Windows::Forms::ToolStrip());
			this->toolStripButton1 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton2 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton3 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton4 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton5 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton6 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton7 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton8 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton9 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton10 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton11 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton12 = (gcnew System::Windows::Forms::ToolStripButton());
			this->toolStripButton13 = (gcnew System::Windows::Forms::ToolStripButton());
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->textBox_X = (gcnew System::Windows::Forms::TextBox());
			this->textBox_Y = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_psize = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_datasize = (gcnew System::Windows::Forms::TextBox());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_MaxSize = (gcnew System::Windows::Forms::TextBox());
			this->button_Clear_Click = (gcnew System::Windows::Forms::Button());
			this->button_Run = (gcnew System::Windows::Forms::Button());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButton_Group = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_Single = (gcnew System::Windows::Forms::RadioButton());
			this->groupBox5 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_CS = (gcnew System::Windows::Forms::ComboBox());
			this->radioButton_NC = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_C2 = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_CS = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton_C1 = (gcnew System::Windows::Forms::RadioButton());
			this->radioButton10 = (gcnew System::Windows::Forms::RadioButton());
			this->groupBox6 = (gcnew System::Windows::Forms::GroupBox());
			this->hScrollBar1 = (gcnew System::Windows::Forms::HScrollBar());
			this->textBox8 = (gcnew System::Windows::Forms::TextBox());
			this->textBox7 = (gcnew System::Windows::Forms::TextBox());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->textBox6 = (gcnew System::Windows::Forms::TextBox());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->textBox5 = (gcnew System::Windows::Forms::TextBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->groupBox7 = (gcnew System::Windows::Forms::GroupBox());
			this->radioButton7 = (gcnew System::Windows::Forms::RadioButton());
			this->textBox_Filename = (gcnew System::Windows::Forms::TextBox());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->richTextBox1 = (gcnew System::Windows::Forms::RichTextBox());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->groupBox8 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_Run = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox9 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_classify = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox10 = (gcnew System::Windows::Forms::GroupBox());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->comboBox_clusters = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox_clustering = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox11 = (gcnew System::Windows::Forms::GroupBox());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->comboBox_NL_degree = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox_regression = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox12 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox13 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_MaxIter = (gcnew System::Windows::Forms::TextBox());
			this->textBox_delta = (gcnew System::Windows::Forms::TextBox());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->groupBox14 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->checkBox_Unbiased = (gcnew System::Windows::Forms::CheckBox());
			this->directoryEntry1 = (gcnew System::DirectoryServices::DirectoryEntry());
			this->groupBox15 = (gcnew System::Windows::Forms::GroupBox());
			this->checkBox_ShowRange = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox16 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_Kmeans_Option = (gcnew System::Windows::Forms::ComboBox());
			this->label19 = (gcnew System::Windows::Forms::Label());
			this->groupBox17 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox_Weight = (gcnew System::Windows::Forms::ComboBox());
			this->label22 = (gcnew System::Windows::Forms::Label());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label21 = (gcnew System::Windows::Forms::Label());
			this->comboBox_kNN = (gcnew System::Windows::Forms::ComboBox());
			this->label20 = (gcnew System::Windows::Forms::Label());
			this->checkBox2 = (gcnew System::Windows::Forms::CheckBox());
			this->groupBox18 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox20 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox_eps = (gcnew System::Windows::Forms::TextBox());
			this->label28 = (gcnew System::Windows::Forms::Label());
			this->groupBox21 = (gcnew System::Windows::Forms::GroupBox());
			this->comboBox5 = (gcnew System::Windows::Forms::ComboBox());
			this->textBox11 = (gcnew System::Windows::Forms::TextBox());
			this->label27 = (gcnew System::Windows::Forms::Label());
			this->comboBox_P_Function = (gcnew System::Windows::Forms::ComboBox());
			this->groupBox19 = (gcnew System::Windows::Forms::GroupBox());
			this->textBox10 = (gcnew System::Windows::Forms::TextBox());
			this->label26 = (gcnew System::Windows::Forms::Label());
			this->label25 = (gcnew System::Windows::Forms::Label());
			this->textBox9 = (gcnew System::Windows::Forms::TextBox());
			this->label24 = (gcnew System::Windows::Forms::Label());
			this->textBox_ini = (gcnew System::Windows::Forms::TextBox());
			this->label23 = (gcnew System::Windows::Forms::Label());
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->imageList1 = (gcnew System::Windows::Forms::ImageList(this->components));
			this->menuStrip1->SuspendLayout();
			this->toolStrip1->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->groupBox4->SuspendLayout();
			this->groupBox5->SuspendLayout();
			this->groupBox6->SuspendLayout();
			this->groupBox7->SuspendLayout();
			this->groupBox8->SuspendLayout();
			this->groupBox9->SuspendLayout();
			this->groupBox10->SuspendLayout();
			this->groupBox11->SuspendLayout();
			this->groupBox12->SuspendLayout();
			this->groupBox13->SuspendLayout();
			this->groupBox14->SuspendLayout();
			this->groupBox15->SuspendLayout();
			this->groupBox16->SuspendLayout();
			this->groupBox17->SuspendLayout();
			this->groupBox18->SuspendLayout();
			this->groupBox20->SuspendLayout();
			this->groupBox21->SuspendLayout();
			this->groupBox19->SuspendLayout();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(5) {
				this->fileToolStripMenuItem,
					this->imagingToolStripMenuItem, this->dataEditToolStripMenuItem, this->aboutToolStripMenuItem, this->applicationsToolStripMenuItem
			});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Padding = System::Windows::Forms::Padding(7, 3, 0, 3);
			this->menuStrip1->Size = System::Drawing::Size(1460, 29);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {
				this->newToolStripMenuItem,
					this->openToolStripMenuItem, this->saveToolStripMenuItem, this->saveAsToolStripMenuItem, this->saveImageAsToolStripMenuItem,
					this->exitToolStripMenuItem
			});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(45, 23);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// newToolStripMenuItem
			// 
			this->newToolStripMenuItem->Name = L"newToolStripMenuItem";
			this->newToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->newToolStripMenuItem->Text = L"New";
			// 
			// openToolStripMenuItem
			// 
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->openToolStripMenuItem->Text = L"Open";
			this->openToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::openToolStripMenuItem_Click);
			// 
			// saveToolStripMenuItem
			// 
			this->saveToolStripMenuItem->Name = L"saveToolStripMenuItem";
			this->saveToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->saveToolStripMenuItem->Text = L"Save";
			this->saveToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveToolStripMenuItem_Click);
			// 
			// saveAsToolStripMenuItem
			// 
			this->saveAsToolStripMenuItem->Name = L"saveAsToolStripMenuItem";
			this->saveAsToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->saveAsToolStripMenuItem->Text = L"Save as";
			this->saveAsToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveAsToolStripMenuItem_Click);
			// 
			// saveImageAsToolStripMenuItem
			// 
			this->saveImageAsToolStripMenuItem->Name = L"saveImageAsToolStripMenuItem";
			this->saveImageAsToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->saveImageAsToolStripMenuItem->Text = L"Save Image as";
			// 
			// exitToolStripMenuItem
			// 
			this->exitToolStripMenuItem->Name = L"exitToolStripMenuItem";
			this->exitToolStripMenuItem->Size = System::Drawing::Size(184, 26);
			this->exitToolStripMenuItem->Text = L"Exit";
			this->exitToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::exitToolStripMenuItem_Click);
			// 
			// imagingToolStripMenuItem
			// 
			this->imagingToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(8) {
				this->showDataToolStripMenuItem1,
					this->clearImageToolStripMenuItem, this->showResultToolStripMenuItem, this->showMeansToolStripMenuItem, this->showContourToolStripMenuItem,
					this->showRegressionToolStripMenuItem, this->showClusteredToolStripMenuItem, this->showClusterCenterToolStripMenuItem1
			});
			this->imagingToolStripMenuItem->Name = L"imagingToolStripMenuItem";
			this->imagingToolStripMenuItem->Size = System::Drawing::Size(80, 23);
			this->imagingToolStripMenuItem->Text = L"Imaging";
			// 
			// showDataToolStripMenuItem1
			// 
			this->showDataToolStripMenuItem1->Name = L"showDataToolStripMenuItem1";
			this->showDataToolStripMenuItem1->Size = System::Drawing::Size(226, 26);
			this->showDataToolStripMenuItem1->Text = L"Show Data";
			this->showDataToolStripMenuItem1->Click += gcnew System::EventHandler(this, &Form1::showDataToolStripMenuItem_Click);
			// 
			// clearImageToolStripMenuItem
			// 
			this->clearImageToolStripMenuItem->Name = L"clearImageToolStripMenuItem";
			this->clearImageToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->clearImageToolStripMenuItem->Text = L"Clear Image";
			this->clearImageToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::clearImageToolStripMenuItem_Click_1);
			// 
			// showResultToolStripMenuItem
			// 
			this->showResultToolStripMenuItem->Name = L"showResultToolStripMenuItem";
			this->showResultToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->showResultToolStripMenuItem->Text = L"Show Result";
			this->showResultToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showResultToolStripMenuItem_Click);
			// 
			// showMeansToolStripMenuItem
			// 
			this->showMeansToolStripMenuItem->Name = L"showMeansToolStripMenuItem";
			this->showMeansToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->showMeansToolStripMenuItem->Text = L"Show Means";
			this->showMeansToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showMeansToolStripMenuItem_Click);
			// 
			// showContourToolStripMenuItem
			// 
			this->showContourToolStripMenuItem->Name = L"showContourToolStripMenuItem";
			this->showContourToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->showContourToolStripMenuItem->Text = L"Show Contour";
			this->showContourToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showContourToolStripMenuItem_Click);
			// 
			// showRegressionToolStripMenuItem
			// 
			this->showRegressionToolStripMenuItem->Name = L"showRegressionToolStripMenuItem";
			this->showRegressionToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->showRegressionToolStripMenuItem->Text = L"Show Regression";
			this->showRegressionToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showRegressionToolStripMenuItem_Click);
			// 
			// showClusteredToolStripMenuItem
			// 
			this->showClusteredToolStripMenuItem->Name = L"showClusteredToolStripMenuItem";
			this->showClusteredToolStripMenuItem->Size = System::Drawing::Size(226, 26);
			this->showClusteredToolStripMenuItem->Text = L"Show Clustered";
			this->showClusteredToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::showClusteredToolStripMenuItem_Click);
			// 
			// showClusterCenterToolStripMenuItem1
			// 
			this->showClusterCenterToolStripMenuItem1->Name = L"showClusterCenterToolStripMenuItem1";
			this->showClusterCenterToolStripMenuItem1->Size = System::Drawing::Size(226, 26);
			this->showClusterCenterToolStripMenuItem1->Text = L"Show Cluster Center";
			this->showClusterCenterToolStripMenuItem1->Click += gcnew System::EventHandler(this, &Form1::showClusterCenterToolStripMenuItem1_Click);
			// 
			// dataEditToolStripMenuItem
			// 
			this->dataEditToolStripMenuItem->Name = L"dataEditToolStripMenuItem";
			this->dataEditToolStripMenuItem->Size = System::Drawing::Size(80, 23);
			this->dataEditToolStripMenuItem->Text = L"DataEdit";
			// 
			// aboutToolStripMenuItem
			// 
			this->aboutToolStripMenuItem->Name = L"aboutToolStripMenuItem";
			this->aboutToolStripMenuItem->Size = System::Drawing::Size(63, 23);
			this->aboutToolStripMenuItem->Text = L"About";
			// 
			// applicationsToolStripMenuItem
			// 
			this->applicationsToolStripMenuItem->Name = L"applicationsToolStripMenuItem";
			this->applicationsToolStripMenuItem->Size = System::Drawing::Size(106, 23);
			this->applicationsToolStripMenuItem->Text = L"Applications";
			// 
			// toolStrip1
			// 
			this->toolStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->toolStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(13) {
				this->toolStripButton1,
					this->toolStripButton2, this->toolStripButton3, this->toolStripButton4, this->toolStripButton5, this->toolStripButton6, this->toolStripButton7,
					this->toolStripButton8, this->toolStripButton9, this->toolStripButton10, this->toolStripButton11, this->toolStripButton12, this->toolStripButton13
			});
			this->toolStrip1->Location = System::Drawing::Point(0, 29);
			this->toolStrip1->Name = L"toolStrip1";
			this->toolStrip1->Size = System::Drawing::Size(1460, 27);
			this->toolStrip1->TabIndex = 1;
			this->toolStrip1->Text = L"toolStrip1";
			// 
			// toolStripButton1
			// 
			this->toolStripButton1->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton1->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton1.Image")));
			this->toolStripButton1->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton1->Name = L"toolStripButton1";
			this->toolStripButton1->Size = System::Drawing::Size(24, 24);
			this->toolStripButton1->Text = L"NEW";
			// 
			// toolStripButton2
			// 
			this->toolStripButton2->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton2->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton2.Image")));
			this->toolStripButton2->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton2->Name = L"toolStripButton2";
			this->toolStripButton2->Size = System::Drawing::Size(24, 24);
			this->toolStripButton2->Text = L"SAVE";
			this->toolStripButton2->Click += gcnew System::EventHandler(this, &Form1::toolStripButton2_Click);
			// 
			// toolStripButton3
			// 
			this->toolStripButton3->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton3->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton3.Image")));
			this->toolStripButton3->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton3->Name = L"toolStripButton3";
			this->toolStripButton3->Size = System::Drawing::Size(24, 24);
			this->toolStripButton3->Text = L"END";
			this->toolStripButton3->Click += gcnew System::EventHandler(this, &Form1::toolStripButton3_Click);
			// 
			// toolStripButton4
			// 
			this->toolStripButton4->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton4->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton4.Image")));
			this->toolStripButton4->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton4->Name = L"toolStripButton4";
			this->toolStripButton4->Size = System::Drawing::Size(24, 24);
			this->toolStripButton4->Text = L"OPEN";
			this->toolStripButton4->Click += gcnew System::EventHandler(this, &Form1::toolStripButton4_Click);
			// 
			// toolStripButton5
			// 
			this->toolStripButton5->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton5->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton5.Image")));
			this->toolStripButton5->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton5->Name = L"toolStripButton5";
			this->toolStripButton5->Size = System::Drawing::Size(24, 24);
			this->toolStripButton5->Text = L"SAVEAS";
			// 
			// toolStripButton6
			// 
			this->toolStripButton6->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton6->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton6.Image")));
			this->toolStripButton6->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton6->Name = L"toolStripButton6";
			this->toolStripButton6->Size = System::Drawing::Size(24, 24);
			this->toolStripButton6->Text = L"SAVEAS2";
			// 
			// toolStripButton7
			// 
			this->toolStripButton7->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton7->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton7.Image")));
			this->toolStripButton7->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton7->Name = L"toolStripButton7";
			this->toolStripButton7->Size = System::Drawing::Size(24, 24);
			this->toolStripButton7->Text = L"UNDO";
			// 
			// toolStripButton8
			// 
			this->toolStripButton8->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton8->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton8.Image")));
			this->toolStripButton8->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton8->Name = L"toolStripButton8";
			this->toolStripButton8->Size = System::Drawing::Size(24, 24);
			this->toolStripButton8->Text = L"REDO";
			// 
			// toolStripButton9
			// 
			this->toolStripButton9->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton9->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton9.Image")));
			this->toolStripButton9->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton9->Name = L"toolStripButton9";
			this->toolStripButton9->Size = System::Drawing::Size(24, 24);
			this->toolStripButton9->Text = L"CUT";
			// 
			// toolStripButton10
			// 
			this->toolStripButton10->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton10->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton10.Image")));
			this->toolStripButton10->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton10->Name = L"toolStripButton10";
			this->toolStripButton10->Size = System::Drawing::Size(24, 24);
			this->toolStripButton10->Text = L"COPY";
			// 
			// toolStripButton11
			// 
			this->toolStripButton11->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton11->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton11.Image")));
			this->toolStripButton11->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton11->Name = L"toolStripButton11";
			this->toolStripButton11->Size = System::Drawing::Size(24, 24);
			this->toolStripButton11->Text = L"PASTE";
			// 
			// toolStripButton12
			// 
			this->toolStripButton12->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton12->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton12.Image")));
			this->toolStripButton12->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton12->Name = L"toolStripButton12";
			this->toolStripButton12->Size = System::Drawing::Size(24, 24);
			this->toolStripButton12->Text = L"toolStripButton12";
			// 
			// toolStripButton13
			// 
			this->toolStripButton13->DisplayStyle = System::Windows::Forms::ToolStripItemDisplayStyle::Image;
			this->toolStripButton13->Image = (cli::safe_cast<System::Drawing::Image^>(resources->GetObject(L"toolStripButton13.Image")));
			this->toolStripButton13->ImageTransparentColor = System::Drawing::Color::Magenta;
			this->toolStripButton13->Name = L"toolStripButton13";
			this->toolStripButton13->Size = System::Drawing::Size(24, 24);
			this->toolStripButton13->Text = L"toolStripButton13";
			// 
			// checkBox1
			// 
			this->checkBox1->AutoSize = true;
			this->checkBox1->Location = System::Drawing::Point(15, 86);
			this->checkBox1->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(106, 23);
			this->checkBox1->TabIndex = 2;
			this->checkBox1->Text = L"Show Data";
			this->checkBox1->UseVisualStyleBackColor = true;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(11, 122);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(22, 19);
			this->label1->TabIndex = 3;
			this->label1->Text = L"X:";
			// 
			// textBox_X
			// 
			this->textBox_X->Location = System::Drawing::Point(38, 117);
			this->textBox_X->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox_X->Name = L"textBox_X";
			this->textBox_X->Size = System::Drawing::Size(94, 27);
			this->textBox_X->TabIndex = 4;
			// 
			// textBox_Y
			// 
			this->textBox_Y->Location = System::Drawing::Point(171, 117);
			this->textBox_Y->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox_Y->Name = L"textBox_Y";
			this->textBox_Y->Size = System::Drawing::Size(94, 27);
			this->textBox_Y->TabIndex = 5;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(144, 122);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(21, 19);
			this->label2->TabIndex = 6;
			this->label2->Text = L"Y:";
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->comboBox_psize);
			this->groupBox1->Location = System::Drawing::Point(272, 72);
			this->groupBox1->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox1->Size = System::Drawing::Size(100, 82);
			this->groupBox1->TabIndex = 7;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"PointSize";
			// 
			// comboBox_psize
			// 
			this->comboBox_psize->FormattingEnabled = true;
			this->comboBox_psize->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"1", L"2", L"3" });
			this->comboBox_psize->Location = System::Drawing::Point(11, 32);
			this->comboBox_psize->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->comboBox_psize->Name = L"comboBox_psize";
			this->comboBox_psize->Size = System::Drawing::Size(82, 27);
			this->comboBox_psize->TabIndex = 0;
			this->comboBox_psize->Text = L"2";
			this->comboBox_psize->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_psize_SelectedIndexChanged);
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->textBox_datasize);
			this->groupBox2->Location = System::Drawing::Point(379, 72);
			this->groupBox2->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox2->Size = System::Drawing::Size(106, 82);
			this->groupBox2->TabIndex = 8;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Data Size";
			// 
			// textBox_datasize
			// 
			this->textBox_datasize->Location = System::Drawing::Point(8, 32);
			this->textBox_datasize->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox_datasize->Name = L"textBox_datasize";
			this->textBox_datasize->Size = System::Drawing::Size(88, 27);
			this->textBox_datasize->TabIndex = 0;
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->textBox_MaxSize);
			this->groupBox3->Location = System::Drawing::Point(492, 72);
			this->groupBox3->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox3->Size = System::Drawing::Size(106, 82);
			this->groupBox3->TabIndex = 9;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"MaxSize";
			// 
			// textBox_MaxSize
			// 
			this->textBox_MaxSize->Location = System::Drawing::Point(8, 32);
			this->textBox_MaxSize->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox_MaxSize->Name = L"textBox_MaxSize";
			this->textBox_MaxSize->Size = System::Drawing::Size(88, 27);
			this->textBox_MaxSize->TabIndex = 0;
			this->textBox_MaxSize->Text = L"3000";
			this->textBox_MaxSize->TextChanged += gcnew System::EventHandler(this, &Form1::textBox_MaxSize_TextChanged);
			// 
			// button_Clear_Click
			// 
			this->button_Clear_Click->Location = System::Drawing::Point(621, 89);
			this->button_Clear_Click->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->button_Clear_Click->Name = L"button_Clear_Click";
			this->button_Clear_Click->Size = System::Drawing::Size(69, 55);
			this->button_Clear_Click->TabIndex = 10;
			this->button_Clear_Click->Text = L"Clear";
			this->button_Clear_Click->UseVisualStyleBackColor = true;
			this->button_Clear_Click->Click += gcnew System::EventHandler(this, &Form1::button_Clear_Click_Click);
			// 
			// button_Run
			// 
			this->button_Run->Location = System::Drawing::Point(701, 89);
			this->button_Run->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->button_Run->Name = L"button_Run";
			this->button_Run->Size = System::Drawing::Size(110, 55);
			this->button_Run->TabIndex = 11;
			this->button_Run->Text = L"Run";
			this->button_Run->UseVisualStyleBackColor = true;
			this->button_Run->Click += gcnew System::EventHandler(this, &Form1::button_Run_Click);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(45, 163);
			this->pictureBox1->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(512, 512);
			this->pictureBox1->TabIndex = 12;
			this->pictureBox1->TabStop = false;
			this->pictureBox1->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseClick);
			this->pictureBox1->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseDown);
			this->pictureBox1->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseMove);
			// 
			// groupBox4
			// 
			this->groupBox4->Controls->Add(this->radioButton_Group);
			this->groupBox4->Controls->Add(this->radioButton_Single);
			this->groupBox4->Location = System::Drawing::Point(602, 163);
			this->groupBox4->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox4->Size = System::Drawing::Size(188, 71);
			this->groupBox4->TabIndex = 13;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"Input Mode";
			// 
			// radioButton_Group
			// 
			this->radioButton_Group->AutoSize = true;
			this->radioButton_Group->Location = System::Drawing::Point(99, 33);
			this->radioButton_Group->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_Group->Name = L"radioButton_Group";
			this->radioButton_Group->Size = System::Drawing::Size(74, 23);
			this->radioButton_Group->TabIndex = 14;
			this->radioButton_Group->Text = L"Group";
			this->radioButton_Group->UseVisualStyleBackColor = true;
			// 
			// radioButton_Single
			// 
			this->radioButton_Single->AutoSize = true;
			this->radioButton_Single->Checked = true;
			this->radioButton_Single->Location = System::Drawing::Point(8, 33);
			this->radioButton_Single->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_Single->Name = L"radioButton_Single";
			this->radioButton_Single->Size = System::Drawing::Size(74, 23);
			this->radioButton_Single->TabIndex = 0;
			this->radioButton_Single->TabStop = true;
			this->radioButton_Single->Text = L"Single";
			this->radioButton_Single->UseVisualStyleBackColor = true;
			// 
			// groupBox5
			// 
			this->groupBox5->Controls->Add(this->comboBox_CS);
			this->groupBox5->Controls->Add(this->radioButton_NC);
			this->groupBox5->Controls->Add(this->radioButton_C2);
			this->groupBox5->Controls->Add(this->radioButton_CS);
			this->groupBox5->Controls->Add(this->radioButton_C1);
			this->groupBox5->Location = System::Drawing::Point(602, 260);
			this->groupBox5->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox5->Name = L"groupBox5";
			this->groupBox5->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox5->Size = System::Drawing::Size(188, 137);
			this->groupBox5->TabIndex = 14;
			this->groupBox5->TabStop = false;
			this->groupBox5->Text = L"Target";
			// 
			// comboBox_CS
			// 
			this->comboBox_CS->FormattingEnabled = true;
			this->comboBox_CS->Location = System::Drawing::Point(99, 62);
			this->comboBox_CS->Name = L"comboBox_CS";
			this->comboBox_CS->Size = System::Drawing::Size(66, 27);
			this->comboBox_CS->TabIndex = 17;
			this->comboBox_CS->Text = L"0";
			this->comboBox_CS->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_CS_SelectedIndexChanged);
			// 
			// radioButton_NC
			// 
			this->radioButton_NC->AutoSize = true;
			this->radioButton_NC->Location = System::Drawing::Point(7, 94);
			this->radioButton_NC->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_NC->Name = L"radioButton_NC";
			this->radioButton_NC->Size = System::Drawing::Size(91, 23);
			this->radioButton_NC->TabIndex = 16;
			this->radioButton_NC->TabStop = true;
			this->radioButton_NC->Text = L"No Class";
			this->radioButton_NC->UseVisualStyleBackColor = true;
			this->radioButton_NC->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_NC_CheckedChanged);
			// 
			// radioButton_C2
			// 
			this->radioButton_C2->AutoSize = true;
			this->radioButton_C2->Location = System::Drawing::Point(7, 62);
			this->radioButton_C2->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_C2->Name = L"radioButton_C2";
			this->radioButton_C2->Size = System::Drawing::Size(75, 23);
			this->radioButton_C2->TabIndex = 15;
			this->radioButton_C2->TabStop = true;
			this->radioButton_C2->Text = L"Class2";
			this->radioButton_C2->UseVisualStyleBackColor = true;
			this->radioButton_C2->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_C2_CheckedChanged);
			// 
			// radioButton_CS
			// 
			this->radioButton_CS->AutoSize = true;
			this->radioButton_CS->Location = System::Drawing::Point(99, 30);
			this->radioButton_CS->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_CS->Name = L"radioButton_CS";
			this->radioButton_CS->Size = System::Drawing::Size(71, 23);
			this->radioButton_CS->TabIndex = 14;
			this->radioButton_CS->TabStop = true;
			this->radioButton_CS->Text = L"Select";
			this->radioButton_CS->UseVisualStyleBackColor = true;
			this->radioButton_CS->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_CS_CheckedChanged);
			// 
			// radioButton_C1
			// 
			this->radioButton_C1->AutoSize = true;
			this->radioButton_C1->Location = System::Drawing::Point(8, 30);
			this->radioButton_C1->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton_C1->Name = L"radioButton_C1";
			this->radioButton_C1->Size = System::Drawing::Size(75, 23);
			this->radioButton_C1->TabIndex = 0;
			this->radioButton_C1->TabStop = true;
			this->radioButton_C1->Text = L"Class1";
			this->radioButton_C1->UseVisualStyleBackColor = true;
			this->radioButton_C1->CheckedChanged += gcnew System::EventHandler(this, &Form1::radioButton_C1_CheckedChanged);
			// 
			// radioButton10
			// 
			this->radioButton10->AutoSize = true;
			this->radioButton10->Location = System::Drawing::Point(36, 30);
			this->radioButton10->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton10->Name = L"radioButton10";
			this->radioButton10->Size = System::Drawing::Size(88, 23);
			this->radioButton10->TabIndex = 0;
			this->radioButton10->TabStop = true;
			this->radioButton10->Text = L"Uniform";
			this->radioButton10->UseVisualStyleBackColor = true;
			// 
			// groupBox6
			// 
			this->groupBox6->Controls->Add(this->hScrollBar1);
			this->groupBox6->Controls->Add(this->textBox8);
			this->groupBox6->Controls->Add(this->textBox7);
			this->groupBox6->Controls->Add(this->label6);
			this->groupBox6->Controls->Add(this->textBox6);
			this->groupBox6->Controls->Add(this->label5);
			this->groupBox6->Controls->Add(this->textBox5);
			this->groupBox6->Controls->Add(this->label4);
			this->groupBox6->Controls->Add(this->label3);
			this->groupBox6->Controls->Add(this->groupBox7);
			this->groupBox6->Location = System::Drawing::Point(602, 426);
			this->groupBox6->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox6->Name = L"groupBox6";
			this->groupBox6->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox6->Size = System::Drawing::Size(188, 322);
			this->groupBox6->TabIndex = 15;
			this->groupBox6->TabStop = false;
			this->groupBox6->Text = L"Group Input";
			// 
			// hScrollBar1
			// 
			this->hScrollBar1->Location = System::Drawing::Point(18, 279);
			this->hScrollBar1->Name = L"hScrollBar1";
			this->hScrollBar1->Size = System::Drawing::Size(151, 21);
			this->hScrollBar1->TabIndex = 16;
			// 
			// textBox8
			// 
			this->textBox8->Location = System::Drawing::Point(112, 236);
			this->textBox8->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox8->Name = L"textBox8";
			this->textBox8->Size = System::Drawing::Size(58, 27);
			this->textBox8->TabIndex = 19;
			this->textBox8->Text = L"0";
			// 
			// textBox7
			// 
			this->textBox7->Location = System::Drawing::Point(112, 201);
			this->textBox7->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox7->Name = L"textBox7";
			this->textBox7->Size = System::Drawing::Size(58, 27);
			this->textBox7->TabIndex = 19;
			this->textBox7->Text = L"100";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(12, 241);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(102, 19);
			this->label6->TabIndex = 18;
			this->label6->Text = L"Rotate Angle:";
			// 
			// textBox6
			// 
			this->textBox6->Location = System::Drawing::Point(112, 167);
			this->textBox6->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox6->Name = L"textBox6";
			this->textBox6->Size = System::Drawing::Size(58, 27);
			this->textBox6->TabIndex = 19;
			this->textBox6->Text = L"100";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(12, 206);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(88, 19);
			this->label5->TabIndex = 18;
			this->label5->Text = L"Range of Y:";
			// 
			// textBox5
			// 
			this->textBox5->Location = System::Drawing::Point(112, 133);
			this->textBox5->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox5->Name = L"textBox5";
			this->textBox5->Size = System::Drawing::Size(58, 27);
			this->textBox5->TabIndex = 17;
			this->textBox5->Text = L"50";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(12, 172);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(89, 19);
			this->label4->TabIndex = 18;
			this->label4->Text = L"Range of X:";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(12, 138);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(80, 19);
			this->label3->TabIndex = 16;
			this->label3->Text = L"# of Point:";
			// 
			// groupBox7
			// 
			this->groupBox7->Controls->Add(this->radioButton7);
			this->groupBox7->Controls->Add(this->radioButton10);
			this->groupBox7->Location = System::Drawing::Point(10, 30);
			this->groupBox7->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox7->Name = L"groupBox7";
			this->groupBox7->Padding = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->groupBox7->Size = System::Drawing::Size(161, 95);
			this->groupBox7->TabIndex = 16;
			this->groupBox7->TabStop = false;
			this->groupBox7->Text = L"Distribution";
			// 
			// radioButton7
			// 
			this->radioButton7->AutoSize = true;
			this->radioButton7->Location = System::Drawing::Point(36, 62);
			this->radioButton7->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->radioButton7->Name = L"radioButton7";
			this->radioButton7->Size = System::Drawing::Size(93, 23);
			this->radioButton7->TabIndex = 1;
			this->radioButton7->TabStop = true;
			this->radioButton7->Text = L"Gaussian";
			this->radioButton7->UseVisualStyleBackColor = true;
			// 
			// textBox_Filename
			// 
			this->textBox_Filename->Location = System::Drawing::Point(1112, 499);
			this->textBox_Filename->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->textBox_Filename->Name = L"textBox_Filename";
			this->textBox_Filename->Size = System::Drawing::Size(255, 27);
			this->textBox_Filename->TabIndex = 21;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(1068, 503);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(36, 19);
			this->label7->TabIndex = 20;
			this->label7->Text = L"File:";
			// 
			// richTextBox1
			// 
			this->richTextBox1->Location = System::Drawing::Point(1072, 534);
			this->richTextBox1->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->richTextBox1->Name = L"richTextBox1";
			this->richTextBox1->Size = System::Drawing::Size(295, 202);
			this->richTextBox1->TabIndex = 22;
			this->richTextBox1->Text = L"";
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(3, 154);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(30, 19);
			this->label8->TabIndex = 23;
			this->label8->Text = L"1.0";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(545, 679);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(30, 19);
			this->label9->TabIndex = 24;
			this->label9->Text = L"1.0";
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Location = System::Drawing::Point(27, 682);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(36, 19);
			this->label10->TabIndex = 25;
			this->label10->Text = L"-1.0";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(3, 660);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(36, 19);
			this->label11->TabIndex = 26;
			this->label11->Text = L"-1.0";
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Location = System::Drawing::Point(288, 682);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(30, 19);
			this->label12->TabIndex = 27;
			this->label12->Text = L"0.0";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(3, 408);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(30, 19);
			this->label13->TabIndex = 28;
			this->label13->Text = L"0.0";
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// groupBox8
			// 
			this->groupBox8->Controls->Add(this->comboBox_Run);
			this->groupBox8->Location = System::Drawing::Point(836, 72);
			this->groupBox8->Name = L"groupBox8";
			this->groupBox8->Size = System::Drawing::Size(184, 84);
			this->groupBox8->TabIndex = 29;
			this->groupBox8->TabStop = false;
			this->groupBox8->Text = L"Run Program";
			// 
			// comboBox_Run
			// 
			this->comboBox_Run->FormattingEnabled = true;
			this->comboBox_Run->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"Classification", L"Clustering", L"Regression" });
			this->comboBox_Run->Location = System::Drawing::Point(14, 33);
			this->comboBox_Run->Name = L"comboBox_Run";
			this->comboBox_Run->Size = System::Drawing::Size(161, 27);
			this->comboBox_Run->TabIndex = 0;
			this->comboBox_Run->Text = L"Classification";
			this->comboBox_Run->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_Run_SelectedIndexChanged);
			// 
			// groupBox9
			// 
			this->groupBox9->Controls->Add(this->comboBox_classify);
			this->groupBox9->Location = System::Drawing::Point(836, 163);
			this->groupBox9->Name = L"groupBox9";
			this->groupBox9->Size = System::Drawing::Size(184, 77);
			this->groupBox9->TabIndex = 30;
			this->groupBox9->TabStop = false;
			this->groupBox9->Text = L"Classification";
			// 
			// comboBox_classify
			// 
			this->comboBox_classify->FormattingEnabled = true;
			this->comboBox_classify->Items->AddRange(gcnew cli::array< System::Object^  >(7) {
				L"Bayes-MAP", L"k-NN", L"Perceptron", L"LVQ",
					L"BPNN", L"LDA", L"SVM-SMO"
			});
			this->comboBox_classify->Location = System::Drawing::Point(14, 32);
			this->comboBox_classify->Name = L"comboBox_classify";
			this->comboBox_classify->Size = System::Drawing::Size(161, 27);
			this->comboBox_classify->TabIndex = 0;
			this->comboBox_classify->Text = L"Bayes-MAP";
			// 
			// groupBox10
			// 
			this->groupBox10->Controls->Add(this->label14);
			this->groupBox10->Controls->Add(this->comboBox_clusters);
			this->groupBox10->Controls->Add(this->comboBox_clustering);
			this->groupBox10->Location = System::Drawing::Point(836, 260);
			this->groupBox10->Name = L"groupBox10";
			this->groupBox10->Size = System::Drawing::Size(253, 73);
			this->groupBox10->TabIndex = 31;
			this->groupBox10->TabStop = false;
			this->groupBox10->Text = L"Clustering";
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(167, 0);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(62, 19);
			this->label14->TabIndex = 2;
			this->label14->Text = L"clusters";
			// 
			// comboBox_clusters
			// 
			this->comboBox_clusters->FormattingEnabled = true;
			this->comboBox_clusters->Location = System::Drawing::Point(171, 29);
			this->comboBox_clusters->Name = L"comboBox_clusters";
			this->comboBox_clusters->Size = System::Drawing::Size(66, 27);
			this->comboBox_clusters->TabIndex = 1;
			this->comboBox_clusters->Text = L"2";
			// 
			// comboBox_clustering
			// 
			this->comboBox_clustering->FormattingEnabled = true;
			this->comboBox_clustering->Items->AddRange(gcnew cli::array< System::Object^  >(5) { L"K-Means", L"FCM", L"EM", L"GUCK", L"EGAC" });
			this->comboBox_clustering->Location = System::Drawing::Point(14, 30);
			this->comboBox_clustering->Name = L"comboBox_clustering";
			this->comboBox_clustering->Size = System::Drawing::Size(129, 27);
			this->comboBox_clustering->TabIndex = 0;
			this->comboBox_clustering->Text = L"K-Means";
			this->comboBox_clustering->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_clustering_SelectedIndexChanged);
			// 
			// groupBox11
			// 
			this->groupBox11->Controls->Add(this->label15);
			this->groupBox11->Controls->Add(this->comboBox_NL_degree);
			this->groupBox11->Controls->Add(this->comboBox_regression);
			this->groupBox11->Location = System::Drawing::Point(836, 354);
			this->groupBox11->Name = L"groupBox11";
			this->groupBox11->Size = System::Drawing::Size(253, 73);
			this->groupBox11->TabIndex = 32;
			this->groupBox11->TabStop = false;
			this->groupBox11->Text = L"Regression";
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(167, 0);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(59, 19);
			this->label15->TabIndex = 2;
			this->label15->Text = L"degree";
			// 
			// comboBox_NL_degree
			// 
			this->comboBox_NL_degree->FormattingEnabled = true;
			this->comboBox_NL_degree->Items->AddRange(gcnew cli::array< System::Object^  >(9) {
				L"2", L"3", L"4", L"5", L"6", L"7", L"8",
					L"9", L"10"
			});
			this->comboBox_NL_degree->Location = System::Drawing::Point(171, 29);
			this->comboBox_NL_degree->Name = L"comboBox_NL_degree";
			this->comboBox_NL_degree->Size = System::Drawing::Size(66, 27);
			this->comboBox_NL_degree->TabIndex = 1;
			this->comboBox_NL_degree->Text = L"2";
			// 
			// comboBox_regression
			// 
			this->comboBox_regression->FormattingEnabled = true;
			this->comboBox_regression->Items->AddRange(gcnew cli::array< System::Object^  >(9) {
				L"Linear", L"Linear-L.n", L"Linear-Log10",
					L"Linear-sat(1/r)", L"Nonlinear", L"k-NN", L"Perceptron", L"Logistic", L"BPNN"
			});
			this->comboBox_regression->Location = System::Drawing::Point(14, 30);
			this->comboBox_regression->Name = L"comboBox_regression";
			this->comboBox_regression->Size = System::Drawing::Size(129, 27);
			this->comboBox_regression->TabIndex = 0;
			this->comboBox_regression->Text = L"Linear";
			this->comboBox_regression->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_regression_SelectedIndexChanged);
			// 
			// groupBox12
			// 
			this->groupBox12->Controls->Add(this->groupBox13);
			this->groupBox12->Location = System::Drawing::Point(836, 441);
			this->groupBox12->Name = L"groupBox12";
			this->groupBox12->Size = System::Drawing::Size(226, 163);
			this->groupBox12->TabIndex = 33;
			this->groupBox12->TabStop = false;
			this->groupBox12->Text = L"General Parameters";
			// 
			// groupBox13
			// 
			this->groupBox13->Controls->Add(this->textBox_MaxIter);
			this->groupBox13->Controls->Add(this->textBox_delta);
			this->groupBox13->Controls->Add(this->label17);
			this->groupBox13->Controls->Add(this->label16);
			this->groupBox13->Location = System::Drawing::Point(14, 27);
			this->groupBox13->Name = L"groupBox13";
			this->groupBox13->Size = System::Drawing::Size(192, 118);
			this->groupBox13->TabIndex = 0;
			this->groupBox13->TabStop = false;
			this->groupBox13->Text = L"Stop Criteria";
			// 
			// textBox_MaxIter
			// 
			this->textBox_MaxIter->Location = System::Drawing::Point(86, 72);
			this->textBox_MaxIter->Name = L"textBox_MaxIter";
			this->textBox_MaxIter->Size = System::Drawing::Size(100, 27);
			this->textBox_MaxIter->TabIndex = 34;
			this->textBox_MaxIter->Text = L"1000";
			// 
			// textBox_delta
			// 
			this->textBox_delta->Location = System::Drawing::Point(86, 31);
			this->textBox_delta->Name = L"textBox_delta";
			this->textBox_delta->Size = System::Drawing::Size(100, 27);
			this->textBox_delta->TabIndex = 2;
			this->textBox_delta->Text = L"1.0e-8";
			// 
			// label17
			// 
			this->label17->AutoSize = true;
			this->label17->Location = System::Drawing::Point(12, 75);
			this->label17->Name = L"label17";
			this->label17->Size = System::Drawing::Size(68, 19);
			this->label17->TabIndex = 1;
			this->label17->Text = L"Max lter:";
			// 
			// label16
			// 
			this->label16->AutoSize = true;
			this->label16->Location = System::Drawing::Point(11, 39);
			this->label16->Name = L"label16";
			this->label16->Size = System::Drawing::Size(47, 19);
			this->label16->TabIndex = 0;
			this->label16->Text = L"delta:";
			// 
			// groupBox14
			// 
			this->groupBox14->Controls->Add(this->textBox3);
			this->groupBox14->Controls->Add(this->label18);
			this->groupBox14->Controls->Add(this->checkBox_Unbiased);
			this->groupBox14->Location = System::Drawing::Point(1042, 168);
			this->groupBox14->Name = L"groupBox14";
			this->groupBox14->Size = System::Drawing::Size(179, 92);
			this->groupBox14->TabIndex = 34;
			this->groupBox14->TabStop = false;
			this->groupBox14->Text = L"Bayes(MAP...)";
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(121, 48);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(40, 27);
			this->textBox3->TabIndex = 35;
			this->textBox3->Text = L"1";
			// 
			// label18
			// 
			this->label18->AutoSize = true;
			this->label18->Location = System::Drawing::Point(10, 54);
			this->label18->Name = L"label18";
			this->label18->Size = System::Drawing::Size(103, 19);
			this->label18->TabIndex = 35;
			this->label18->Text = L"Ellipse sigma:";
			// 
			// checkBox_Unbiased
			// 
			this->checkBox_Unbiased->AutoSize = true;
			this->checkBox_Unbiased->Checked = true;
			this->checkBox_Unbiased->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox_Unbiased->Location = System::Drawing::Point(14, 26);
			this->checkBox_Unbiased->Name = L"checkBox_Unbiased";
			this->checkBox_Unbiased->Size = System::Drawing::Size(97, 23);
			this->checkBox_Unbiased->TabIndex = 0;
			this->checkBox_Unbiased->Text = L"Unbiased";
			this->checkBox_Unbiased->UseVisualStyleBackColor = true;
			// 
			// groupBox15
			// 
			this->groupBox15->Controls->Add(this->checkBox_ShowRange);
			this->groupBox15->Location = System::Drawing::Point(836, 611);
			this->groupBox15->Name = L"groupBox15";
			this->groupBox15->Size = System::Drawing::Size(165, 52);
			this->groupBox15->TabIndex = 35;
			this->groupBox15->TabStop = false;
			this->groupBox15->Text = L"Clustered Range";
			// 
			// checkBox_ShowRange
			// 
			this->checkBox_ShowRange->AutoSize = true;
			this->checkBox_ShowRange->Checked = true;
			this->checkBox_ShowRange->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox_ShowRange->Location = System::Drawing::Point(24, 26);
			this->checkBox_ShowRange->Name = L"checkBox_ShowRange";
			this->checkBox_ShowRange->Size = System::Drawing::Size(70, 23);
			this->checkBox_ShowRange->TabIndex = 0;
			this->checkBox_ShowRange->Text = L"Show";
			this->checkBox_ShowRange->UseVisualStyleBackColor = true;
			// 
			// groupBox16
			// 
			this->groupBox16->Controls->Add(this->comboBox_Kmeans_Option);
			this->groupBox16->Controls->Add(this->label19);
			this->groupBox16->Location = System::Drawing::Point(836, 679);
			this->groupBox16->Name = L"groupBox16";
			this->groupBox16->Size = System::Drawing::Size(217, 59);
			this->groupBox16->TabIndex = 36;
			this->groupBox16->TabStop = false;
			this->groupBox16->Text = L"K-Means(FCM,EM)";
			// 
			// comboBox_Kmeans_Option
			// 
			this->comboBox_Kmeans_Option->FormattingEnabled = true;
			this->comboBox_Kmeans_Option->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"Original", L"Furthest Point", L"k-means++" });
			this->comboBox_Kmeans_Option->Location = System::Drawing::Point(78, 20);
			this->comboBox_Kmeans_Option->Name = L"comboBox_Kmeans_Option";
			this->comboBox_Kmeans_Option->Size = System::Drawing::Size(121, 27);
			this->comboBox_Kmeans_Option->TabIndex = 1;
			this->comboBox_Kmeans_Option->Text = L"Original";
			// 
			// label19
			// 
			this->label19->AutoSize = true;
			this->label19->Location = System::Drawing::Point(5, 23);
			this->label19->Name = L"label19";
			this->label19->Size = System::Drawing::Size(67, 19);
			this->label19->TabIndex = 0;
			this->label19->Text = L"Initial K: ";
			// 
			// groupBox17
			// 
			this->groupBox17->Controls->Add(this->comboBox_Weight);
			this->groupBox17->Controls->Add(this->label22);
			this->groupBox17->Controls->Add(this->textBox1);
			this->groupBox17->Controls->Add(this->label21);
			this->groupBox17->Controls->Add(this->comboBox_kNN);
			this->groupBox17->Controls->Add(this->label20);
			this->groupBox17->Location = System::Drawing::Point(1042, 72);
			this->groupBox17->Name = L"groupBox17";
			this->groupBox17->Size = System::Drawing::Size(200, 90);
			this->groupBox17->TabIndex = 37;
			this->groupBox17->TabStop = false;
			this->groupBox17->Text = L"k-NN";
			// 
			// comboBox_Weight
			// 
			this->comboBox_Weight->FormattingEnabled = true;
			this->comboBox_Weight->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"Average", L"1/Dist", L"RBF" });
			this->comboBox_Weight->Location = System::Drawing::Point(92, 55);
			this->comboBox_Weight->Name = L"comboBox_Weight";
			this->comboBox_Weight->Size = System::Drawing::Size(102, 27);
			this->comboBox_Weight->TabIndex = 5;
			this->comboBox_Weight->Text = L"Average";
			// 
			// label22
			// 
			this->label22->AutoSize = true;
			this->label22->Location = System::Drawing::Point(7, 62);
			this->label22->Name = L"label22";
			this->label22->Size = System::Drawing::Size(78, 19);
			this->label22->TabIndex = 4;
			this->label22->Text = L"weighted:";
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(139, 22);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(55, 27);
			this->textBox1->TabIndex = 3;
			this->textBox1->Text = L"0.01";
			// 
			// label21
			// 
			this->label21->AutoSize = true;
			this->label21->Location = System::Drawing::Point(92, 27);
			this->label21->Name = L"label21";
			this->label21->Size = System::Drawing::Size(47, 19);
			this->label21->TabIndex = 2;
			this->label21->Text = L"s^2=";
			// 
			// comboBox_kNN
			// 
			this->comboBox_kNN->FormattingEnabled = true;
			this->comboBox_kNN->Items->AddRange(gcnew cli::array< System::Object^  >(30) {
				L"1", L"3", L"5", L"7", L"9", L"11", L"13",
					L"15", L"17", L"19", L"21", L"23", L"25", L"27", L"29", L"31", L"33", L"35", L"37", L"39", L"41", L"43", L"45", L"47", L"49",
					L"51", L"53", L"55", L"57", L"59"
			});
			this->comboBox_kNN->Location = System::Drawing::Point(34, 22);
			this->comboBox_kNN->Name = L"comboBox_kNN";
			this->comboBox_kNN->Size = System::Drawing::Size(52, 27);
			this->comboBox_kNN->TabIndex = 1;
			this->comboBox_kNN->Text = L"1";
			this->comboBox_kNN->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::comboBox_kNN_SelectedIndexChanged);
			// 
			// label20
			// 
			this->label20->AutoSize = true;
			this->label20->Location = System::Drawing::Point(7, 27);
			this->label20->Name = L"label20";
			this->label20->Size = System::Drawing::Size(20, 19);
			this->label20->TabIndex = 0;
			this->label20->Text = L"k:";
			// 
			// checkBox2
			// 
			this->checkBox2->AutoSize = true;
			this->checkBox2->Checked = true;
			this->checkBox2->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox2->Location = System::Drawing::Point(128, 86);
			this->checkBox2->Name = L"checkBox2";
			this->checkBox2->Size = System::Drawing::Size(129, 23);
			this->checkBox2->TabIndex = 38;
			this->checkBox2->Text = L"Fixed Contour";
			this->checkBox2->UseVisualStyleBackColor = true;
			// 
			// groupBox18
			// 
			this->groupBox18->Controls->Add(this->groupBox20);
			this->groupBox18->Controls->Add(this->groupBox19);
			this->groupBox18->Location = System::Drawing::Point(1248, 72);
			this->groupBox18->Name = L"groupBox18";
			this->groupBox18->Size = System::Drawing::Size(186, 400);
			this->groupBox18->TabIndex = 39;
			this->groupBox18->TabStop = false;
			this->groupBox18->Text = L"Neural Networks";
			// 
			// groupBox20
			// 
			this->groupBox20->Controls->Add(this->textBox_eps);
			this->groupBox20->Controls->Add(this->label28);
			this->groupBox20->Controls->Add(this->groupBox21);
			this->groupBox20->Controls->Add(this->comboBox_P_Function);
			this->groupBox20->Location = System::Drawing::Point(11, 213);
			this->groupBox20->Name = L"groupBox20";
			this->groupBox20->Size = System::Drawing::Size(163, 179);
			this->groupBox20->TabIndex = 1;
			this->groupBox20->TabStop = false;
			this->groupBox20->Text = L"Perceptron";
			// 
			// textBox_eps
			// 
			this->textBox_eps->Location = System::Drawing::Point(77, 54);
			this->textBox_eps->Name = L"textBox_eps";
			this->textBox_eps->Size = System::Drawing::Size(70, 27);
			this->textBox_eps->TabIndex = 3;
			this->textBox_eps->Text = L"0.001";
			// 
			// label28
			// 
			this->label28->AutoSize = true;
			this->label28->Location = System::Drawing::Point(11, 60);
			this->label28->Name = L"label28";
			this->label28->Size = System::Drawing::Size(62, 19);
			this->label28->TabIndex = 2;
			this->label28->Text = L"Epsilon:";
			// 
			// groupBox21
			// 
			this->groupBox21->Controls->Add(this->comboBox5);
			this->groupBox21->Controls->Add(this->textBox11);
			this->groupBox21->Controls->Add(this->label27);
			this->groupBox21->Location = System::Drawing::Point(8, 89);
			this->groupBox21->Name = L"groupBox21";
			this->groupBox21->Size = System::Drawing::Size(146, 82);
			this->groupBox21->TabIndex = 1;
			this->groupBox21->TabStop = false;
			this->groupBox21->Text = L"BP";
			// 
			// comboBox5
			// 
			this->comboBox5->FormattingEnabled = true;
			this->comboBox5->Location = System::Drawing::Point(6, 49);
			this->comboBox5->Name = L"comboBox5";
			this->comboBox5->Size = System::Drawing::Size(112, 27);
			this->comboBox5->TabIndex = 2;
			this->comboBox5->Text = L"sigmoid";
			// 
			// textBox11
			// 
			this->textBox11->Location = System::Drawing::Point(65, 18);
			this->textBox11->Name = L"textBox11";
			this->textBox11->Size = System::Drawing::Size(53, 27);
			this->textBox11->TabIndex = 1;
			this->textBox11->Text = L"5";
			// 
			// label27
			// 
			this->label27->AutoSize = true;
			this->label27->Location = System::Drawing::Point(61, 0);
			this->label27->Name = L"label27";
			this->label27->Size = System::Drawing::Size(64, 19);
			this->label27->TabIndex = 0;
			this->label27->Text = L"Hidden:";
			// 
			// comboBox_P_Function
			// 
			this->comboBox_P_Function->FormattingEnabled = true;
			this->comboBox_P_Function->Items->AddRange(gcnew cli::array< System::Object^  >(4) { L"hardlims", L"linear", L"sigmoid", L"tanh()" });
			this->comboBox_P_Function->Location = System::Drawing::Point(15, 21);
			this->comboBox_P_Function->Name = L"comboBox_P_Function";
			this->comboBox_P_Function->Size = System::Drawing::Size(121, 27);
			this->comboBox_P_Function->TabIndex = 0;
			this->comboBox_P_Function->Text = L"hardlims";
			// 
			// groupBox19
			// 
			this->groupBox19->Controls->Add(this->textBox10);
			this->groupBox19->Controls->Add(this->label26);
			this->groupBox19->Controls->Add(this->label25);
			this->groupBox19->Controls->Add(this->textBox9);
			this->groupBox19->Controls->Add(this->label24);
			this->groupBox19->Controls->Add(this->textBox_ini);
			this->groupBox19->Controls->Add(this->label23);
			this->groupBox19->Controls->Add(this->comboBox1);
			this->groupBox19->Location = System::Drawing::Point(12, 23);
			this->groupBox19->Name = L"groupBox19";
			this->groupBox19->Size = System::Drawing::Size(162, 185);
			this->groupBox19->TabIndex = 0;
			this->groupBox19->TabStop = false;
			this->groupBox19->Text = L"Learning Rate";
			// 
			// textBox10
			// 
			this->textBox10->Location = System::Drawing::Point(63, 140);
			this->textBox10->Name = L"textBox10";
			this->textBox10->Size = System::Drawing::Size(65, 27);
			this->textBox10->TabIndex = 7;
			this->textBox10->Text = L"5";
			// 
			// label26
			// 
			this->label26->AutoSize = true;
			this->label26->Location = System::Drawing::Point(7, 162);
			this->label26->Name = L"label26";
			this->label26->Size = System::Drawing::Size(63, 19);
			this->label26->TabIndex = 6;
			this->label26->Text = L"b/(b+n)";
			// 
			// label25
			// 
			this->label25->AutoSize = true;
			this->label25->Location = System::Drawing::Point(36, 140);
			this->label25->Name = L"label25";
			this->label25->Size = System::Drawing::Size(21, 19);
			this->label25->TabIndex = 5;
			this->label25->Text = L"b:";
			// 
			// textBox9
			// 
			this->textBox9->Location = System::Drawing::Point(63, 101);
			this->textBox9->Name = L"textBox9";
			this->textBox9->Size = System::Drawing::Size(65, 27);
			this->textBox9->TabIndex = 4;
			this->textBox9->Text = L"0.95";
			// 
			// label24
			// 
			this->label24->AutoSize = true;
			this->label24->Location = System::Drawing::Point(6, 104);
			this->label24->Name = L"label24";
			this->label24->Size = System::Drawing::Size(51, 19);
			this->label24->TabIndex = 3;
			this->label24->Text = L"DF/M:";
			// 
			// textBox_ini
			// 
			this->textBox_ini->Location = System::Drawing::Point(63, 66);
			this->textBox_ini->Name = L"textBox_ini";
			this->textBox_ini->Size = System::Drawing::Size(65, 27);
			this->textBox_ini->TabIndex = 2;
			this->textBox_ini->Text = L"0.5";
			// 
			// label23
			// 
			this->label23->AutoSize = true;
			this->label23->Location = System::Drawing::Point(6, 69);
			this->label23->Name = L"label23";
			this->label23->Size = System::Drawing::Size(50, 19);
			this->label23->TabIndex = 1;
			this->label23->Text = L"Initial:";
			// 
			// comboBox1
			// 
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Location = System::Drawing::Point(14, 30);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(132, 27);
			this->comboBox1->TabIndex = 0;
			this->comboBox1->Text = L"Specified";
			// 
			// imageList1
			// 
			this->imageList1->ColorDepth = System::Windows::Forms::ColorDepth::Depth8Bit;
			this->imageList1->ImageSize = System::Drawing::Size(16, 16);
			this->imageList1->TransparentColor = System::Drawing::Color::Transparent;
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(9, 19);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1460, 770);
			this->Controls->Add(this->groupBox18);
			this->Controls->Add(this->checkBox2);
			this->Controls->Add(this->groupBox17);
			this->Controls->Add(this->groupBox16);
			this->Controls->Add(this->groupBox15);
			this->Controls->Add(this->groupBox14);
			this->Controls->Add(this->groupBox12);
			this->Controls->Add(this->groupBox11);
			this->Controls->Add(this->groupBox10);
			this->Controls->Add(this->groupBox9);
			this->Controls->Add(this->groupBox8);
			this->Controls->Add(this->label13);
			this->Controls->Add(this->label12);
			this->Controls->Add(this->label11);
			this->Controls->Add(this->label10);
			this->Controls->Add(this->label9);
			this->Controls->Add(this->label8);
			this->Controls->Add(this->richTextBox1);
			this->Controls->Add(this->textBox_Filename);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->groupBox6);
			this->Controls->Add(this->groupBox5);
			this->Controls->Add(this->groupBox4);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->button_Run);
			this->Controls->Add(this->button_Clear_Click);
			this->Controls->Add(this->groupBox3);
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->textBox_Y);
			this->Controls->Add(this->textBox_X);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->checkBox1);
			this->Controls->Add(this->toolStrip1);
			this->Controls->Add(this->menuStrip1);
			this->Font = (gcnew System::Drawing::Font(L"微軟正黑體", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(136)));
			this->MainMenuStrip = this->menuStrip1;
			this->Margin = System::Windows::Forms::Padding(3, 4, 3, 4);
			this->Name = L"Form1";
			this->Text = L"Machine Learning (u10216005)";
			this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->toolStrip1->ResumeLayout(false);
			this->toolStrip1->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			this->groupBox5->ResumeLayout(false);
			this->groupBox5->PerformLayout();
			this->groupBox6->ResumeLayout(false);
			this->groupBox6->PerformLayout();
			this->groupBox7->ResumeLayout(false);
			this->groupBox7->PerformLayout();
			this->groupBox8->ResumeLayout(false);
			this->groupBox9->ResumeLayout(false);
			this->groupBox10->ResumeLayout(false);
			this->groupBox10->PerformLayout();
			this->groupBox11->ResumeLayout(false);
			this->groupBox11->PerformLayout();
			this->groupBox12->ResumeLayout(false);
			this->groupBox13->ResumeLayout(false);
			this->groupBox13->PerformLayout();
			this->groupBox14->ResumeLayout(false);
			this->groupBox14->PerformLayout();
			this->groupBox15->ResumeLayout(false);
			this->groupBox15->PerformLayout();
			this->groupBox16->ResumeLayout(false);
			this->groupBox16->PerformLayout();
			this->groupBox17->ResumeLayout(false);
			this->groupBox17->PerformLayout();
			this->groupBox18->ResumeLayout(false);
			this->groupBox20->ResumeLayout(false);
			this->groupBox20->PerformLayout();
			this->groupBox21->ResumeLayout(false);
			this->groupBox21->PerformLayout();
			this->groupBox19->ResumeLayout(false);
			this->groupBox19->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	//Form1_Load
	private: System::Void Form1_Load(System::Object^  sender, System::EventArgs^  e) {
		myBitmap = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height, PixelFormat::Format24bppRgb);
		g = Graphics::FromImage(myBitmap);
		button_Clear_Click_Click(sender, e);
		Pi = 4.0 * atan(1.0);
		imW = pictureBox1->Width;
		imH = pictureBox1->Height;
		CenterX=0.5*imW; //Center X座標
		CenterY=0.5*imH; //Center Y座標
		comboBox_psize->SelectedIndex=1; //PointSize=2，預設資料點大小。
		PointSize=(comboBox_psize->SelectedIndex+1)*5;
		PointSize1=PointSize+2; //資料點運算完結果大小
		PointSize2=PointSize+4; //Class or Cluster 中心大小
		radioButton_C1->Checked=true; //ClassKind=1; Color=Red
		ClassKind=1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
		//comboBox_CS Items initializing, i.e. Class(Target) selection.
		for (int i=0;i<10;i++)
			comboBox_CS->Items->Add(Convert::ToString(i));
		comboBox_CS->SelectedIndex=0;
		comboBox_CS->Enabled=false;

		HandFlag=true; //HandFlag=true時，PictureBox1_MouseClick可輸入Data Points。反之，則不能。
		
		totalCTestData = imW * imH;
		totalRTestData = imW;

		MaxSizeOfData=Convert::ToInt32(textBox_MaxSize->Text); //MaxSize Of Input Data
		InputData = new pData[MaxSizeOfData];
		NewPublicVariables(MaxSizeOfData);
		for (int i = 0; i < totalCTestData; i++) {
			for (int j = 0; j < 255; j++) {
				ALLNNs[i][j] = 0;
			}
		}
		//Run Program
		comboBox_Run->SelectedIndex=0; //Run Classification--1:Clustering--2:Regression
		comboBox_clustering->Enabled=false; //Disable Clustering
		comboBox_clusters->Enabled=false; //Disable Clustering
		comboBox_regression->Enabled=false; //Disable Regression
		//classification Method
		comboBox_classify->SelectedIndex=0;
		
		//Group Input
		//srand( (unsigned)time(NULL) );
		//Distribution=1; //Distribution=1 --> Gaussian , Distribution=0 --> Uniform
		//NumberOfPoint=Convert::ToInt32(textBox_Num_p->Text); //Number Of Point per Click
		//RangeX=Convert::ToInt32(textBox_R_X->Text); //Range of X
		//RangeY=Convert::ToInt32(textBox_R_Y->Text); //Range of Y
		
		//Regression
		comboBox_NL_degree->Enabled=false; //Nonlinear Regression only use.

		//clustering Method
		STOPFlag=true;
		comboBox_clustering->SelectedIndex=0;
		comboBox_Kmeans_Option->SelectedIndex=0;
		for (int i=0; i<30; i++)
			comboBox_clusters->Items->Add(Convert::ToString(i+2));
		comboBox_clusters->SelectedIndex=0;
		NumOfCluster= comboBox_clusters->SelectedIndex+2;
		
	}//Form1_Load
	
	//pictureBox_Mouse
	private: System::Void pictureBox1_MouseMove(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		//textBox_X->Text =Convert::ToString((e->X-256.0)/256.0);
		//textBox_Y->Text =Convert::ToString((256.0-e->Y)/256.0);
		textBox_X->Text =Convert::ToString((e->X-CenterX)/CenterX);
		textBox_Y->Text =Convert::ToString((CenterY-e->Y)/CenterY);
	}
	private: System::Void pictureBox1_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		X_Cur=e->X;
		Y_Cur=e->Y;
	}
	private: System::Void pictureBox1_MouseClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		double X_tmp=0.0;
		double Y_tmp=0.0;
		if (HandFlag) { //HandFlag=true時，PictureBox1_MouseClick可輸入Data Points。反之，則不能。
			if (radioButton_Single->Checked) {
				//X_tmp=(double)(X_Cur-256.0)/256.0;
				//Y_tmp=(double)(256.0-Y_Cur)/256.0;
				X_tmp=(double)(X_Cur-CenterX)/CenterX;
				Y_tmp=(double)(CenterY-Y_Cur)/CenterY;
				InputData[NumberOfData].X=X_tmp;
				InputData[NumberOfData].Y=Y_tmp;
				if (radioButton_C1->Checked){
					InputData[NumberOfData].ClassKind=1; //Red Color
				}//if (radioButton_C1->Checked)
				else if (radioButton_C2->Checked){
					InputData[NumberOfData].ClassKind=-1; //Blue Color
				}//else if (radioButton_C2->Checked)
				else if (radioButton_NC->Checked){
					InputData[NumberOfData].ClassKind=-2; //Black Color
				}//else if (radioButton_NC->Checked)
				else{
					InputData[NumberOfData].ClassKind=comboBox_CS->SelectedIndex; //Select Class Color
				}//else
				bshDraw=ClassToColor(InputData[NumberOfData].ClassKind);
				g->FillEllipse(bshDraw, X_Cur-PointSize/2, Y_Cur-PointSize/2, PointSize, PointSize);
				NumberOfData++;
				textBox_datasize->Text = Convert::ToString(NumberOfData);
			}//if single input
			else { //group input
				MessageBox::Show("建構中，請耐心等候。");
			}//else Group input
		}//if (HandFlag)
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//pictureBox1_MouseClick
	
	//Clear
	private: System::Void button_Clear_Click_Click(System::Object^  sender, System::EventArgs^  e) {
		clearImageToolStripMenuItem_Click(sender, e);
		NumberOfData=0;
		textBox_datasize->Text = "0";
		Filename1="";
		textBox_Filename->Text ="";
		richTextBox1->Clear();
	}
	private: System::Void clearImageToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		g->Clear(Color::White); //Paint over the image area in white.
		//bshDraw = gcnew SolidBrush(Color::White); //清除畫布另一種方法
		//g->FillRectangle(bshDraw, 0, 0, 512, 512);
		//Draw
		penDraw = gcnew Pen(Color::Black, 1); //畫筆是黑色的, 畫框
		//g->DrawLine(penDraw, 0, 0, 0, 511);
		//g->DrawLine(penDraw, 0, 0, 511, 0);
		//g->DrawLine(penDraw, 0, 511, 511, 511);
		//g->DrawLine(penDraw, 511, 0, 511, 511);
		g->DrawLine(penDraw, 0, 0, 0, pictureBox1->Height-1);
		g->DrawLine(penDraw, 0, 0, pictureBox1->Width-1, 0);
		g->DrawLine(penDraw, 0, pictureBox1->Height-1, pictureBox1->Width-1, pictureBox1->Height-1);
		g->DrawLine(penDraw, pictureBox1->Width-1, 0, pictureBox1->Width-1, pictureBox1->Height-1);
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//clearImageToolStripMenuItem
	private: System::Void clearImageToolStripMenuItem_Click_1(System::Object^  sender, System::EventArgs^  e) {
		clearImageToolStripMenuItem_Click(sender, e);
	}
	
	//Exit
	private: System::Void exitToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		delete g;
		delete [] InputData;
		DeletePublicVariables(MaxSizeOfData);
		Application::Exit();
	}	
	private: System::Void toolStripButton3_Click(System::Object^  sender, System::EventArgs^  e) {
		exitToolStripMenuItem_Click(sender, e);
	}
	
	//Save
	private: System::Void saveToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if (String::IsNullOrEmpty(Filename1))
			saveAsToolStripMenuItem_Click(sender, e);
		else
			richTextBox1->SaveFile(Filename1);
	}
	private: System::Void toolStripButton2_Click(System::Object^  sender, System::EventArgs^  e) {
		saveToolStripMenuItem_Click(sender, e);
	}
	
	//Save As
	private: System::Void saveAsToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		if (NumberOfData>0) {
			richTextBox1->Clear();
			for (int i=0; i<NumberOfData; i++) {
				String^ line = Convert::ToString(InputData[i].X) + "\t" + Convert::ToString(InputData[i].Y) + "\t"+ Convert::ToString(InputData[i].ClassKind)+ "\n";
				richTextBox1->AppendText(line);
			}//for i
			saveFileDialog1->Filter="*.TXT|*.txt|*.DAT|*.dat|All Files|*.*";
			//saveFileDialog1->DefaultExt="dat";
			saveFileDialog1->DefaultExt = "txt";
			if (saveFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK && saveFileDialog1->FileName->Length>0) {
				richTextBox1->SaveFile(saveFileDialog1->FileName);
				Filename1 = saveFileDialog1->FileName;
				textBox_Filename->Text = saveFileDialog1->FileName;
			}//if saveFileDialog1
		}//if (NumberOfData>0)
	}
	
	//Open
	private: System::Void openToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		bool openfile = true;
		if (richTextBox1->Modified) { //判斷 richTextBox1是否有經過任何編輯
			bool openfile = false;
		if (MessageBox::Show("未存檔!是否繼續", "確認視窗", MessageBoxButtons::YesNo, MessageBoxIcon::Question)==System::Windows::Forms::DialogResult::Yes)
			openfile = true;
		}//if (richTextBox1->Modified)
		if (openfile) {
			openFileDialog1->Filter="*.TXT|*.txt|*.DAT|*.dat|All Files|*.*";
			if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
				button_Clear_Click_Click(sender, e);
				richTextBox1->LoadFile(openFileDialog1->FileName);
				Filename1 = openFileDialog1->FileName;
				textBox_Filename->Text = openFileDialog1->SafeFileName;
				NumberOfData=0;
				//char* Token =(char*)(void*)Marshal::StringToHGlobalAnsi(richTextBox1->Text);
				char* Token=(char*)Marshal::StringToHGlobalAnsi(richTextBox1->Text).ToPointer();
				char* sptoken = strtok(Token," \t\n");
				while (sptoken != NULL) {
					InputData[NumberOfData].X= atof(sptoken);
					sptoken = strtok(NULL," \t\n");
					InputData[NumberOfData].Y= atof(sptoken);
					sptoken = strtok(NULL," \t\n");
					InputData[NumberOfData].ClassKind= atoi(sptoken);
					sptoken = strtok(NULL," \t\n");
					NumberOfData++;
				}//while
				textBox_datasize->Text = Convert::ToString(NumberOfData);
				showDataToolStripMenuItem_Click(sender, e); // // Show data
				//Backup ClassKindfor clustering
				for (int j=0; j<NumberOfData; j++)
					BackupClassKind[j]=InputData[j].ClassKind;
			}//if openFileDialog1
		}//if (openfile)
	}
	private: System::Void toolStripButton4_Click(System::Object^  sender, System::EventArgs^  e) {
		openToolStripMenuItem_Click(sender, e);
	}
	
	//Show Data_menuStrip
	private: System::Void showDataToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		clearImageToolStripMenuItem_Click(sender, e);
		//DrawData
		for (int i=0; i<NumberOfData; i++) {
			//X_Cur=(int)(InputData[i].X*256+256);
			//Y_Cur=(int)(256-InputData[i].Y*256);
			X_Cur=(int)(InputData[i].X*CenterX+CenterX);
			Y_Cur=(int)(CenterY-InputData[i].Y*CenterY);
			bshDraw=ClassToColor(InputData[i].ClassKind);
			g->FillEllipse(bshDraw, X_Cur - PointSize / 2, Y_Cur - PointSize / 2, PointSize, PointSize);
		}// for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}
	
	//Target_Class1
	private: System::Void radioButton_C1_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if (radioButton_C1->Checked){
			ClassKind=1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
		}//if
	}
	//Target_Class2
	private: System::Void radioButton_C2_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if (radioButton_C2->Checked){
			ClassKind=-1; //ClassKind=1; Color=Red, ClassKind=-1; Color=Blue
		}//if
	}
	//Target_No Class
	private: System::Void radioButton_NC_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		if (radioButton_NC->Checked){
			ClassKind=-2; //ClassKind=-2; Color=Black
		}//if
	}
	//Target_Selected
	private: System::Void radioButton_CS_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
		comboBox_CS->Enabled=radioButton_CS->Checked;
		if (radioButton_CS->Checked){
			ClassKind=comboBox_CS->SelectedIndex;
		}//if
	}
	private: System::Void comboBox_CS_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		if (radioButton_CS->Checked){
			ClassKind=comboBox_CS->SelectedIndex;
		}//if
	}

	//Point Size
	private: System::Void comboBox_psize_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		PointSize=(comboBox_psize->SelectedIndex+1)*5;
		PointSize1=PointSize+2;
		PointSize2=PointSize+4;
	}
	
	//Max Size
	private: System::Void textBox_MaxSize_TextChanged(System::Object^  sender, System::EventArgs^  e) {
		MaxSizeOfData=Convert::ToInt32(textBox_MaxSize->Text);
		if (MaxSizeOfData<1000){
			MaxSizeOfData=1000;
			textBox_MaxSize->Text=Convert::ToString(MaxSizeOfData);
		}//if
		else {
			//DeletePublicVariables(MaxSizeOfData);
			delete [] InputData;
			//NewPublicVariables(MaxSizeOfData);
			InputData = new pData[MaxSizeOfData];
		}//else
		button_Clear_Click_Click(sender, e);
	}
	
	//Run_Btn
	private: System::Void button_Run_Click(System::Object^  sender, System::EventArgs^  e) {
		int tF;
		MethodCodeValue = 0;
		int methodbased = 20;
		switch (comboBox_Run->SelectedIndex) {
			case 0: //Classification
				MethodCodeValue = (comboBox_Run->SelectedIndex)*methodbased+ comboBox_classify->SelectedIndex;
				break;
			case 1: //Clustering
				MethodCodeValue = (comboBox_Run->SelectedIndex)*methodbased+ comboBox_clustering->SelectedIndex;
				break;
			case 2: //Regression
				MethodCodeValue = (comboBox_Run->SelectedIndex)*methodbased+ comboBox_regression->SelectedIndex;
				break;
			default:
				MessageBox::Show("無建構此類別方法!");
		}//switch ---Run Method-----Based = 20
		switch (MethodCodeValue) {
			case 0: //Classification--Bayes-MAP
				BayesMAP();
				showContourToolStripMenuItem_Click(sender, e);
				showResultToolStripMenuItem_Click(sender, e);
				showMeansToolStripMenuItem_Click(sender, e);
				break;
			case 1: // Classification—k-NN
				if (MaxKNN>0) {
					Create_kNN_Contour_Table();
					showContourToolStripMenuItem_Click(sender, e);
					showResultToolStripMenuItem_Click(sender, e);
				}//if
				break;
			case 2: //Perceptron Classification
				comboBox_P_Function->SelectedIndex = 0; //指定Transfer Function = hardlims()
				Perceptron_CTrain(); //Training Weights
				showContourToolStripMenuItem_Click(sender, e);
				showResultToolStripMenuItem_Click(sender, e);
				break;
			case 20: //Clustering--K-Means
				NumOfClusters=comboBox_clusters->SelectedIndex+2;
				K_Means(NumOfClusters);
				clearImageToolStripMenuItem_Click(sender, e);
				showClusteredToolStripMenuItem_Click(sender, e);
				showClusterCenterToolStripMenuItem1_Click(sender, e);
				break;
			case 21: //Clustering—Fuzzy C-Means
				NumOfClusters = comboBox_clusters->SelectedIndex + 2;
				FCM(NumOfClusters);
				clearImageToolStripMenuItem_Click(sender, e);
				showClusteredToolStripMenuItem_Click(sender, e);
				showClusterCenterToolStripMenuItem1_Click(sender, e);
				break;
			case 40: //Regression--Linear
				LinearRegression();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 41: //Regression--Linear-ln() == log e
				LinearRegressionLn();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 44: //Nonlinear Regression--Degree == NLdegree
				NLdegree= comboBox_NL_degree->SelectedIndex+2;
				A=new double*[NLdegree+1];
				for (int i=0; i<NLdegree+1; i++)
					A[i]= new double[NLdegree+1];
				B=new double[NLdegree+1];
				NLcoef=new double[NLdegree+1];
				NonlinearRegression(NLdegree);
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				//delete
				for (int i=0; i<NLdegree+1; i++)
					delete [] A[i];
				delete [] A;
				delete [] B;
				delete [] NLcoef;
				break;
			case 45: // Regression—k-NN
				kNNs = comboBox_kNN->SelectedIndex * 2 + 1;
				BuildAllkNNRegTable();
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			case 46: //Perceptron Regression
				if (comboBox_P_Function->SelectedIndex == 0)//若Transfer Function = hardlims()，則指定成linear()
					comboBox_P_Function->SelectedIndex = 1; //因Transfer Function須為連續函數才可微分。
				tF = comboBox_P_Function->SelectedIndex;
				Perceptron_RTrain(tF); //Training Weights
				showDataToolStripMenuItem_Click(sender, e);
				showRegressionToolStripMenuItem_Click(sender, e);
				break;
			default:
				MessageBox::Show("無建構此類別方法!");
		}//switch (MethodCodeValue)
	}//button_Run_Click
	
	//Show Result_menuStrip
	private: System::Void showResultToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		double Output_yi;
		Brush^ errBrush= gcnew SolidBrush(Color::DeepPink); //Classified error color
		//Draw Data
		for (int i = 0; i<NumberOfData; i++) {
			X_Cur= (int)(InputData[i].X*CenterX+ CenterX); //CenterX=256.0
			Y_Cur= (int)(CenterY-InputData[i].Y*CenterY); //CenterY=256.0
			switch (MethodCodeValue) {
			case 0: //Bayesian-MAP
				Output_yi= PClass1*PxyClass1[i] -PClass2*PxyClass2[i];
				if (Sgn((double)InputData[i].ClassKind) == Sgn(Output_yi)) {
					bshDraw= ClassToColor(InputData[i].ClassKind);
					g->FillEllipse(bshDraw, X_Cur-PointSize1 / 2, Y_Cur-PointSize1 / 2, PointSize1, PointSize1);
				}//if (Sgn((double) InputData[i].ClassKind)== Sgn(Output_yi))
				else {
					g->FillEllipse(errBrush, X_Cur-PointSize1 / 2, Y_Cur-PointSize1 / 2, PointSize1, PointSize1);
				}//else
				break;
			case 1: //K-NN
				bshDraw = ClassToColor(InputData[i].ClassKind);
				g->FillEllipse(bshDraw, X_Cur - PointSize1 / 2, Y_Cur - PointSize1 / 2, PointSize1, PointSize1);
				break;
			case 2: // Perceptron Classification
				Output_yi = (double)InputData[i].ClassKind*PerceptronClassify(InputData[i]);
				if (Sgn(Output_yi)>0) {
					bshDraw = ClassToColor(InputData[i].ClassKind);
					g->FillEllipse(bshDraw, X_Cur - PointSize1 / 2, Y_Cur - PointSize1 / 2, PointSize1, PointSize1);
				}//if
				else {
					g->FillEllipse(errBrush, X_Cur - PointSize1 / 2, Y_Cur - PointSize1 / 2, PointSize1, PointSize1);
				}//else
				break;
			default:
				MessageBox::Show("無建構此類別方法!");
			}//switch
		}//for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showResultToolStripMenuItem_Click
	
	//Show Means_menuStrip
	private: System::Void showMeansToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		//Draw Means 1
		X_Cur = (int)(MeanX1*CenterX + CenterX); //CenterX=256.0
		Y_Cur= (int)(CenterY-MeanY1*CenterY); //CenterY=256.0
		//bshDraw=ClassToColor(1); //Red
		//g->FillEllipse(bshDraw, X_Cur-PointSize2/2, Y_Cur-PointSize2/2, PointSize2, PointSize2);
		penDraw= ClassToPenColor(1); //Red
		g->DrawEllipse(penDraw, X_Cur-PointSize2 / 2, Y_Cur-PointSize2 / 2, PointSize2, PointSize2);
		//Draw Means 2
		X_Cur= (int)(MeanX2*CenterX+ CenterX);
		Y_Cur= (int)(CenterY-MeanY2*CenterY);
		//bshDraw=ClassToColor(-1); //Blue
		//g->FillEllipse(bshDraw, X_Cur-PointSize2/2, Y_Cur-PointSize2/2, PointSize2, PointSize2);
		penDraw= ClassToPenColor(-1); //Blue
		g->DrawEllipse(penDraw, X_Cur-PointSize2 / 2, Y_Cur-PointSize2 / 2, PointSize2, PointSize2);
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showMeansToolStripMenuItem_Click
	
	//Show Contour_menuStrip
	private: System::Void showContourToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		int i;
		double Output_yi;
		pData Sample;
		unsigned char blockcolor=0, LightLevel, ColorScale;
		int Lighttmp;
		//DrawData
		// Lock the bitmap's bits.
		Rectangle rect= Rectangle(0, 0, imW, imH);
		BitmapData^ bmpData= myBitmap->LockBits(rect, ImageLockMode::ReadWrite, myBitmap->PixelFormat);
		int ByteOfSkip= bmpData->Stride -bmpData->Width * 3;//計算每行後面幾個Padding bytes , 全彩影像
		unsigned char* p = (unsigned char*)bmpData->Scan0.ToPointer();
		int index = 0;
		for (int y = 0; y < imH; y++){
			for (int x = 0; x < imW; x++){
				Sample.X = (double)(x -CenterX) / CenterX;
				Sample.Y = (double)(CenterY-y) / CenterY;
				switch (MethodCodeValue) {
				case 0: //Bayesian-MAP
					ColorScale= 200;
					Output_yi= PClass1*evalPxy1(Sample) -PClass2*evalPxy2(Sample);
					break;
				case 1: //K-NN
					i = (unsigned int)y*imW + x;
					Output_yi = Sgn((double)(ALLCountClass1[i] - ALLCountClass2[i]));
					break;
				case 2: //Perceptron Classification
					ColorScale = 1;
					Output_yi = PerceptronClassify(Sample);
					break;
				default:
					MessageBox::Show("無建構此類別方法!");
				}//switch
				//Show Contour Type
				LightLevel= 155; //Fixed or Soft --> LightLevel=Min(155, (int)abs(255.0* Output_yi* ColorScale) );
				if (Output_yi>0.0) {
					p[index + 0] = LightLevel; //Red
					p[index + 1] = LightLevel; //Green
					p[index + 2] = 255 -blockcolor; //Blue
				}//if
				else if (Output_yi== 0.0) {
					p[index + 0] = 255; //Red
					p[index + 1] = 255; //Green
					p[index + 2] = 255; //Blue
				}//else if (Output_yi==0)
				else {
					p[index + 0] = 255 -blockcolor; //Red
					p[index + 1] = LightLevel; //Green
					p[index + 2] = LightLevel; //Blue
				}//else
				index += 3;
			}//for x
			index += ByteOfSkip; // 跳過剩下的Padding bytes
		}//for y
		// Unlock the bits.
		myBitmap->UnlockBits(bmpData);
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showContourToolStripMenuItem_Click
	
	//Run Program_comboBox
	private: System::Void comboBox_Run_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		switch (comboBox_Run->SelectedIndex) {
			case 0: //Classification
				comboBox_classify->Enabled=true; //Enable Classification
				comboBox_clustering->Enabled=false; //Disable Clustering
				comboBox_clusters->Enabled=false; //Disable Clustering
				comboBox_regression->Enabled=false; //Disable Regression
				break;
			case 1: //Clustering
				comboBox_clustering->Enabled=true; //Enable Clustering
				comboBox_clusters->Enabled=true; //Disable Clustering
				break;
			case 2: //Regression
				comboBox_regression->Enabled=true; //EnableRegression
				break;
			default: //Classification
				comboBox_classify->Enabled=true; //Enable Classification
				comboBox_clustering->Enabled=false; //Disable Clustering
				comboBox_clusters->Enabled=false; //Disable Clustering
				comboBox_regression->Enabled=false; //Disable Regression
		}//switch
	}//comboBox_Run_SelectedIndexChanged
	
	//Regression_comboBox
	private: System::Void comboBox_regression_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		showDataToolStripMenuItem_Click(sender, e);
		if (comboBox_regression->SelectedIndex==4)
			comboBox_NL_degree->Enabled=true; //Nonlinear Regression only use.
		else
			comboBox_NL_degree->Enabled=false; //Nonlinear Regression only use.
	}

	//showRegression_menuStrip
	private: System::Void showRegressionToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		double LowBound, HighBound, tmpX, tmpY, wk, sumW, s2;
		int X0, Y0;
		int tF = comboBox_P_Function->SelectedIndex;
		LowBound=-1.0; HighBound=1.0;
		X0=0; //tmpX=-1.0 , i.e. shiftX=1.0
		switch (comboBox_regression->SelectedIndex) {
		case 0: //Linear
			tmpY= LR_a0 -LR_a1; //tmpX=-1.0
			break;
		case 1: //Linear--Ln
			tmpY= LR_a0*exp(LR_a1); //tmpX=-1.0 , i.e. shiftX=1.0, ==>Y = a0 * exp^(a1*1.0)
			LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
			break;
		case 2: //Linear—Log10
			break;
		case 4: //Nonlinear
			//tmpY=NLcoef[0]-NLcoef[1];
			//for (inti=2; i<NLdegree+1;i++)
			// tmpY += NLcoef[i]*pow(-1.0,i); //tmpX=-1.0
			tmpY=NLcoef[NLdegree];
			for (int i=NLdegree-1; i>=0;i--)
			tmpY= -tmpY+NLcoef[i]; //tmpX=-1.0
			break;
		case 5: //k-NN
			tmpY = 0.0; sumW = 0.0; //tmpX=-1.0
			switch (comboBox_Weight->SelectedIndex){
			case 0: // Y=1/K * Sum_kNNs(InputData[].Y)
				for (int i = 0; i < kNNs; i++)
					tmpY += InputData[NNs[X0][i]].Y;
				tmpY /= kNNs;
				//MessageBox::Show("" + NNs[X0][200]);
				break;
			case 1: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=1/dist_k
				break;
			case 2: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=RBF(dist_k)
				break;
			}//switch(comboBox_Weight->SelectedIndex)
			break;
		case 6:
			tmpY = PerceptronRegression(-1.0, tF);
			break;
		default:
			tmpY= LR_a0-LR_a1; //tmpX=-1.0
		}//switch
		Y0=(int)((HighBound-tmpY)*CenterY);
		while (Y0<1 || Y0 >imH-2 ) {
			X0++;
			switch (comboBox_regression->SelectedIndex) {
			case 0: //Linear
				tmpX= (double) (X0-CenterX)/CenterX;
				tmpY= LR_a0 + LR_a1*tmpX;
				break;
			case 1: //Linear--Ln
				tmpX= (double) (X0+CenterX)/CenterX;
				tmpY= LR_a0*exp(LR_a1*tmpX);
				LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
				break;
			case 4: //Nonlinear
				tmpX = (double) (X0-CenterX)/CenterX; //tmpX=[-1.0,1.0]
				//tmpY=NLcoef[0];
				//for (inti=1; i<NLdegree+1;i++)
				// tmpY+= NLcoef[i]*pow(tmpX,i);
				tmpY=NLcoef[NLdegree];
				for (int i=NLdegree-1; i>=0;i--)
					tmpY= tmpY*tmpX+NLcoef[i];
				break;
			case 5: //k-NN
				break;
			case 6:
				tmpX = (double)(X0 - CenterX) / CenterX;
				tmpY = PerceptronRegression(tmpX, tF);
				break;
			default:
				tmpY= LR_a0 + LR_a1*tmpX;
			}//switch
			Y0=(int)((HighBound-tmpY)*CenterY);
		}//while
		for (int x = X0+1; x < imW-1; x++){
			switch (comboBox_regression->SelectedIndex) {
				case 0: //Linear
					tmpX= (double) (x-CenterX)/CenterX;
					tmpY = LR_a0 + LR_a1 * tmpX; //Y = a0 + a1*X
					break;
				case 1: //Linear--Ln
					tmpX= (double) (x+CenterX)/CenterX;
					tmpY = LR_a0*exp(LR_a1*tmpX); //Y = a0 * exp^(a1*X)
					LowBound=1.0; HighBound=3.0; //CenterX=256.0 (==2.0) ==>[1.0,3.0]
					break;
				case 4: //Nonlinear
					tmpX= (double) (x-CenterX)/CenterX; //tmpX=[-1.0,1.0]
					//tmpY=NLcoef[0];
					//for (inti=1; i<NLdegree+1;i++)
					//tmpY+= NLcoef[i]*pow(tmpX,i);
					tmpY=NLcoef[NLdegree];
					for (int i=NLdegree-1; i>=0;i--)
						tmpY= tmpY*tmpX+NLcoef[i];
					break;
				case 5: //k-NN
					tmpY = 0.0; sumW = 0.0; //tmpX=-1.0
					switch (comboBox_Weight->SelectedIndex){
						case 0: // Y=1/K * Sum_kNNs(InputData[].Y)
							for (int i = 0; i < kNNs; i++)
								tmpY += InputData[NNs[x][i]].Y;
							tmpY /= kNNs;
							break;
						case 1: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=1/dist_k

							break;
						case 2: // Y=1/sumW* Sum_kNNs(InputData[].Y * W) W=RBF(dist_k)

							break;
					}//switch(comboBox_Weight->SelectedIndex)
					break;
				case 6:
					tmpX = (double)(X0 - CenterX) / CenterX;
					tmpY = PerceptronRegression(tmpX, tF);
					break;
				default: //Linear
					tmpX= (double) (x-CenterX)/CenterX;
					tmpY = LR_a0 + LR_a1 * tmpX; //Y = a0 + a1*X
			}//switch
			if (tmpY> LowBound&& tmpY< HighBound) {
				X_Cur=x;
				Y_Cur=(int)((HighBound-tmpY)*CenterY); //CenterY=256.0
				penDraw=ClassToPenColor(-1); //Blue
				g->DrawLine(penDraw, X0, Y0, X_Cur, Y_Cur);
				X0=X_Cur;
				Y0=Y_Cur;
			}//if
		}//for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showRegressionToolStripMenuItem_Click

	//Clustering_comboBox
	private: System::Void comboBox_clustering_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		//Return ClassKindto InputData
		for (int j=0; j<NumberOfData; j++)
			InputData[j].ClassKind=BackupClassKind[j];
		showDataToolStripMenuItem_Click(sender, e);	
	}

	//Show Clustered_menuStrip
	private: System::Void showClusteredToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		//Draw Clustered Data
		int r=PointSize/2;
		for (int i=0; i<NumberOfData; i++) {
			X_Cur=(int)(InputData[i].X*CenterX+CenterX); //CenterX=256.0
			Y_Cur=(int)(CenterY-InputData[i].Y*CenterY); //CenterY=256.0
			penDraw=ClassToPenColor(InputData[i].ClassKind);
			switch (InputData[i].ClassKind) {
				case 0: //Circle
					g->DrawEllipse(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
					break;
				case 1: //X
					g->DrawLine(penDraw, X_Cur-r, Y_Cur-r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur+r, Y_Cur-r);
					break;
				case 2: //Triangle
					g->DrawLine(penDraw, X_Cur, Y_Cur-r, X_Cur-r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur+r, X_Cur, Y_Cur-r);
					break;
				case 3: //Rectangle
					g->DrawRectangle(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
					break;
				case 4: //菱形
					g->DrawLine(penDraw, X_Cur, Y_Cur-r, X_Cur+r, Y_Cur);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur, X_Cur, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur, Y_Cur+r, X_Cur-r, Y_Cur);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur, X_Cur, Y_Cur-r);
					break;
				case 5: //梯形
					g->DrawLine(penDraw, X_Cur-r+1, Y_Cur-r, X_Cur+r-1, Y_Cur-r);
					g->DrawLine(penDraw, X_Cur+r-1, Y_Cur-r, X_Cur+r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur+r, Y_Cur+r, X_Cur-r, Y_Cur+r);
					g->DrawLine(penDraw, X_Cur-r, Y_Cur+r, X_Cur-r+1, Y_Cur-r);
					break;
				default:
					g->DrawEllipse(penDraw, X_Cur-r, Y_Cur-r, PointSize, PointSize);
			}//switch
		}// for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showClusteredToolStripMenuItem_Click

	//Show Cluster Center_menuStrip
	private: System::Void showClusterCenterToolStripMenuItem1_Click(System::Object^  sender, System::EventArgs^  e) {
		int disttmp, j;
		if (checkBox_ShowRange->Checked) {
			//Compute the radius of each cluster
			for (int k=0; k<NumOfClusters; k++)
				Radius[k]= 0;
			for (int i=0; i<NumberOfData; i++) {
				switch (comboBox_clustering->SelectedIndex) {
					case 0: //k-Means
					case 1: //FCM
						disttmp=(int) (sqrt(dist[i][0])*CenterX+0.5);
						if (disttmp > Radius[InputData[i].ClassKind])
							Radius[InputData[i].ClassKind]= disttmp;
						break;
					case 2: //EM
					case 3: //FuzzyGG
						break;
					default: //k-Means
						disttmp=(int) (sqrt(dist[i][0])*CenterX+0.5);
						if (disttmp > Radius[InputData[i].ClassKind])
							Radius[InputData[i].ClassKind]= disttmp;
				}//switch
			} //for i
		}//if
		for (int i=0; i<NumOfClusters; i++) {
			X_Cur=(int)(ClusterCenter[i].X*CenterX+CenterX); //CenterX=256.0
			Y_Cur=(int)(CenterY-ClusterCenter[i].Y*CenterY); //CenterY=256.0
			bshDraw=ClassToColor(ClusterCenter[i].ClassKind);
			g->FillEllipse(bshDraw, X_Cur-PointSize2/2, Y_Cur-PointSize2/2, PointSize2, PointSize2);
			if (checkBox_ShowRange->Checked) {
				penDraw=ClassToPenColor(ClusterCenter[i].ClassKind);
				//(顏色，圓左上角x座標，圓左上角Y座標，X軸直徑，Y軸直徑)
				g->DrawEllipse(penDraw, X_Cur-Radius[i], Y_Cur-Radius[i], 2*Radius[i], 2*Radius[i]);
			}//if
		}//for
		pictureBox1->Image = myBitmap;
		pictureBox1->Refresh();
	}//showClusterCenterToolStripMenuItem_Click
	
	//kNN_comboBox
	private: System::Void comboBox_kNN_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
		int i;
		MaxKNN = FindMaxKNN();
		kNNs = comboBox_kNN->SelectedIndex * 2 + 1;
		if (comboBox_classify->SelectedIndex == 1) {
			if (kNNs>MaxKNN) { //if (kNNs>Number Of Classification Data) set kNNs=3.
				comboBox_kNN->SelectedIndex = 0; //if (kNNs>Number Of Classification Data) set kNNs=3.
				kNNs = 1;
				MessageBox::Show("k太大，已超過MaxKNN最大鄰居數量! 請重選。");
			}//if
			else {
				//Count Neighbors Class Type.
				for (int y = 0; y < imH; y++) {
					for (int x = 0; x < imW; x++) {
						i = y*imW + x;
						ALLCountClass1[i] = 0; ALLCountClass2[i] = 0;
						for (int j = 0; j<kNNs; j++) {
							if (InputData[ALLNNs[i][j]].ClassKind == 1)
								ALLCountClass1[i]++;
							else
								ALLCountClass2[i]++;
						}//for j
					}//for x
				}//for y
			}//else
		}//if(comboBox_classify->SelectedIndex==1)
		else if (kNNs>NumberOfData - 1 && comboBox_regression->SelectedIndex == 5) {
			comboBox_kNN->SelectedIndex = 0; //if (kNNs>Number Of regression Data) set kNNs=1.
			kNNs = 1;
			MessageBox::Show("k太大，已超過Data最大鄰居數量! 請重選。");
		}//else if
	}//comboBox_kNN_SelectedIndexChanged

};
}