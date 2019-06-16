

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QPixmap pix;    // 画板
    QPoint lastPoint;
    QPoint endPoint;
    QColor penColor;

    QImage input_img;  // 原图原版
    QPixmap input_pix; //原图缩放版
    QPixmap seed_pix;   // 种子图
    QPixmap result_pix; // 结果图
    QSize originSize;   // 原大小
    int penWidth;
    bool isDrawing;   //标志是否正在绘图
    int *input_array = NULL;    // 原图int*
    int *mask_array = NULL;     // 遮罩int*
    int draw_method;
    bool isGPU = true;

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
