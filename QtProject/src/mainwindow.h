// mainwindow头函数
// 声明mainwindow类及类内变量
// 声明CUDA内定义的、需要在QT代码中调用的函数

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

// cuda中定义的函数，分别为cut操作(GPU)、cut操作(CPU)、锐化、模糊、风格化、直方图均衡
// 具体参数意义及实现详见.cu文件
extern "C"
int* getCutMask(int *src_img, int *mask_img, int img_height, int img_width);
extern "C"
int* getCutMask_iSAP(int *src_img, int *mask_img, int img_height, int img_width);
extern "C"
int* imgSharpen(int *src_img, int img_height, int img_width);
extern "C"
int* imgBlur(int *src_img, int img_height, int img_width);
extern "C"
int* imgCLAHE(int *src_img, int img_height, int img_width);
extern "C"
int* imgCLAHE_Global(int *src_img, int img_height, int img_width);


#define INIT_WIDTH 20
#define ADD_COLOR (0, 255, 0)
#define DEL_COLOR (255, 0, 0)
#define WINDOW_SIZE 700
#define BOUNDING 10

namespace Ui {
class MainWindow;
}

// 主窗口类定义：
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QPixmap pix;    // 画板
    // 笔触参数
    QPoint lastPoint;
    QPoint endPoint;
    QColor penColor;    //笔触颜色
    int penWidth;       //笔触宽度
    bool isDrawing;   //标志是否正在绘图

    // 图片、画布、数组变量保存
    QImage input_img;  // 原图原版
    QPixmap input_pix; //原图缩放版
    QPixmap seed_pix;   // 种子图
    QPixmap result_pix; // 结果图
    QSize originSize;   // 原大小
    int *input_array = NULL;    // 原图int*
    int *mask_array = NULL;     // 遮罩int*

    // 其它标识
    int draw_method;    // 渲染区域：0：全图；1：前景；2：背景；3：笔触覆盖
    bool isGPU = true;  // 是否用GPU进行Cut操作

protected:
    void paintEvent(QPaintEvent *);
    void mousePressEvent(QMouseEvent *);
    void mouseMoveEvent(QMouseEvent *);
    void mouseReleaseEvent(QMouseEvent *);
    void process_pic(int* (*pf)(int*,int, int));
private slots:
    void on_pushButton_clicked();
    void on_pushButton_2_clicked();
    void on_horizontalSlider_valueChanged(int value);
    void on_pushButton_3_clicked();
    void on_pushButton_4_clicked();
    void on_pushButton_5_clicked();
    void on_checkBox_stateChanged(int arg1);
    void on_pushButton_6_clicked();
    void on_pushButton_7_clicked();
    void on_pushButton_9_clicked();
    void on_pushButton_8_clicked();
    void on_pushButton_10_clicked();
    void on_pushButton_11_clicked();
    void on_comboBox_activated(int index);
};

#endif // MAINWINDOW_H
