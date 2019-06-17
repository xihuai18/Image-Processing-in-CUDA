// 主窗口实现文件
// 绑定UI控件信息槽，窗口绘制
// 实现画笔交互
// 调用CUDA进行图像处理，将处理结果转换为图像
#include <QPainter>
#include <QMouseEvent>
#include <QPushButton>
#include <QDebug>
#include <QFileDialog>
#include <QSize>
#include <stack>
#include <QBuffer>
#include <fstream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<ctime>



// 主窗口初始化：初始化默认参数
// parent: 父布局
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // 笔触是否落下
    isDrawing = false;
    // 设置笔触开始点（画布之外）
    lastPoint.setX(-INIT_WIDTH-BOUNDING);
    lastPoint.setY(-INIT_WIDTH-BOUNDING);
    endPoint.setX(-INIT_WIDTH-BOUNDING);
    endPoint.setY(-INIT_WIDTH-BOUNDING);
    // 初始化笔触宽度、颜色
    penWidth = INIT_WIDTH;
    penColor = QColor DEL_COLOR;
    // 渲染范围默认为全图
    draw_method = 0;
}

// 主窗口析构函数：删除ui及两个（若未删除）数组
MainWindow::~MainWindow()
{
    delete ui;
    if (input_array != nullptr)
        delete []input_array;
    if (mask_array != nullptr)
        delete []mask_array;
}

// 绘制动作：定义UI每次被调用update()时的绘制行为：
// event: 绘制动作
void MainWindow::paintEvent(QPaintEvent *event)
{
    // 获取笔触参数
      QPen pen;
      pen.setWidth(penWidth);
      pen.setColor(penColor);
      QPainter painter(this);
      // 绘制两个画布
      painter.drawPixmap(BOUNDING, BOUNDING, input_pix);// 绘制操作区
      painter.drawPixmap(2*BOUNDING+WINDOW_SIZE,BOUNDING, result_pix);// 绘制结果区

      if(isDrawing) //如果正在绘图，就在辅助画布上绘制
      {
         //将以前pix中的内容复制到tempPix中，保证以前的内容不消失
         QPainter pp(&pix);
         pp.setPen(pen);
         pp.drawLine(lastPoint, endPoint);    // 让前一个坐标值等于后一个坐标值，这样就能实现画出连续的线
         painter.drawPixmap(BOUNDING, BOUNDING, pix);
         // 更新开始点为当前结束点
         lastPoint = endPoint;
      }
      else{
          // 不做改变，保持绘制
          painter.drawPixmap(BOUNDING, BOUNDING, pix);
      }

}

// 鼠标点击事件：设置绘制状态，记录开始点
// event：鼠标事件
void MainWindow::mousePressEvent(QMouseEvent *event)
{
               if(event->button()==Qt::LeftButton) //鼠标左键按下
               {
                   lastPoint = event->pos();
                   endPoint = event->pos();
                   isDrawing = true;   //正在绘图

               }
}

// 鼠标移动事件：设置绘制状态，若持续在按着左键拖动鼠标，则连续绘制结束点
// event：鼠标事件
void MainWindow::mouseMoveEvent(QMouseEvent *event)
{

         if(event->buttons()&Qt::LeftButton) //鼠标左键按下的同时移动鼠标
        {
                  endPoint = event->pos();
                  update(); //进行绘制
         }
}

// 鼠标释放事件：松开鼠标，更新结束点，停止绘制状态
// event：鼠标事件
void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{

            if(event->button() == Qt::LeftButton) //鼠标左键释放
             {
                      endPoint = event->pos();
                      update();
                      isDrawing = false;    //结束绘图
             }
}

// 笔触颜色设置：设置笔触为“删除”
void MainWindow::on_pushButton_clicked()
{
    penColor = QColor DEL_COLOR;
}

// 笔触颜色设置：设置笔触为“保留”
void MainWindow::on_pushButton_2_clicked()
{
    penColor = QColor ADD_COLOR;
}

// 滑块事件：通过滑块改变笔触粗细
// value：笔触width
void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    penWidth = value;
}

// 保存图片：对图像处理完成后，将图像保存至硬盘。
void MainWindow::on_pushButton_3_clicked()
{
    // 获取保存位置
    QString fileName = QFileDialog::getSaveFileName(this, tr("Open File"), ".", tr("Images (*.png *.bmp *.jpg)"));
    // 图像变回原始大小
    QPixmap save_pix = result_pix.scaled(originSize);
    bool flag = save_pix.save(fileName);
    // 保存成功日志
    qDebug()<<"SAVE"<<flag<<"SAVE WHERE"<<fileName;
}

// 加载图片：将图片从硬盘加载入操作区
void MainWindow::on_pushButton_4_clicked()
{
    // 图片路径获取
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), ".", tr("Images (*.png *.bmp *.jpg)"));
    // 图片读取
    QPixmap tmp;
    bool flag = tmp.load(fileName);
    originSize = tmp.size();
    // 图片缩放至窗口大小，交给input_pix显示
    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    input_pix = tmp.scaled(qSize,Qt::KeepAspectRatio);
    qDebug()<<fileName<<flag<<qSize<<tmp.size()<<input_pix.size();

    //画布初始化
    pix = QPixmap(input_pix.size());
    pix.fill(Qt::transparent);//用透明色填充
    // 结果画布以原图为起点
    result_pix = input_pix.copy();

    // 将输入图片保存为数组，以便后续处理
    input_img = tmp.toImage();
    if (input_array != nullptr)
        delete []input_array;
    input_array = new int[input_img.height() * input_img.width()];
    for (int i=0; i<input_img.height(); i++){
        for (int j=0; j<input_img.width(); j++){
            input_array[j*input_img.height() + i] = (input_img.pixelColor(j, i).red()<<16)
                    + (input_img.pixelColor(j, i).green()<<8)
                    + (input_img.pixelColor(j, i).blue());

        }
    }
}

// 裁剪图片：根据画布上的笔触，运行Onecut算法进行前景、背景分离，并在结果图中透明化背景
void MainWindow::on_pushButton_5_clicked()
{
    // 种子数据获取
    seed_pix = pix.scaled(originSize);
    QImage seed_img = seed_pix.toImage();
    int *seed_array = new int[seed_img.height() * seed_img.width()];
    for (int j=0; j<seed_img.width(); j++){
        for (int i=0; i<seed_img.height(); i++){
            seed_array[j*seed_img.height() + i] = (seed_img.pixelColor(j, i).red()<<16)
                    + (seed_img.pixelColor(j, i).green()<<8)
                    + seed_img.pixelColor(j, i).blue();
        }
    }

    // 对算法进行计时
    std::clock_t startTime,endTime;
    startTime = std::clock();

    if (mask_array != nullptr)
        delete []mask_array;
    // GPU算法
    if (isGPU)
        mask_array = getCutMask(input_array, seed_array, seed_img.width(), seed_img.height());
    // CPU算法
    else
        mask_array = getCutMask_iSAP(input_array, seed_array, seed_img.width(), seed_img.height());

    // 计时结束，打印日志
    endTime = std::clock();
    qDebug()<<"CUT FINISH!"<< "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" ;

    // 对分割结果进行显示——背景透明化
    QImage result_img = input_img.copy();
    result_img = result_img.convertToFormat(QImage::Format_ARGB32);
    for (int j=0; j<seed_img.width(); j++){
        for (int i=0; i<seed_img.height(); i++){
            if (mask_array[j*seed_img.height() + i] == 255){
                result_img.setPixelColor(j, i,  Qt::transparent);
            }
        }
    }

    // 结果画布绘制
    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    result_pix.fill(Qt::transparent);
    result_pix = QPixmap::fromImage( result_img.scaled(qSize, Qt::KeepAspectRatio) );
    update();

    delete []seed_array;
}

// 设置Onecut的运行设备（checkbox事件）
// gpuflag: 0为未勾选（CPU），2为已勾选（GPU）
void MainWindow::on_checkBox_stateChanged(int gpuflag)
{
    qDebug()<<"is GPU? "<<gpuflag;
    if (gpuflag == 0){
        isGPU = false;
    }
    else{
        isGPU = true;
    }
}

// 图片处理统一接口
// pf：图像处理函数（processin function）指针
void MainWindow::process_pic(int* (*pf)(int*,int, int)){
    // 种子信息获取（为局部信息处理预备）
    seed_pix = pix.scaled(originSize);
    QImage seed_img = seed_pix.toImage();
    int *seed_array = new int[seed_img.height() * seed_img.width()];
    for (int j=0; j<seed_img.width(); j++){
        for (int i=0; i<seed_img.height(); i++){
            seed_array[j*seed_img.height() + i] = (seed_img.pixelColor(j, i).red()<<16)
                    + (seed_img.pixelColor(j, i).green()<<8)
                    + seed_img.pixelColor(j, i).blue();
        }
    }

    // 开始计时
    std::clock_t startTime,endTime;
    startTime = std::clock();
    // 调用CUDA函数进行图像处理
    int * result_array = pf(input_array, input_img.width(), input_img.height());
    // 计时结束，打印时间
    endTime = std::clock();
    qDebug()<<"FINISH!"<< "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" ;

    // 处理结果绘制
    QImage result_img = input_img.copy();
    for (int j=0; j<input_img.width(); j++){
        for (int i=0; i<input_img.height(); i++){
            // 若原图此处透明则跳过
            if (input_img.pixelColor(j, i) != Qt::transparent){
                // 对应不同的操作范围作不同的处理，若在操作范围内则将原图替换为处理后的结果
                if (draw_method==0 || mask_array && draw_method==1 && mask_array[j*input_img.height() + i]!=255 ||
                        mask_array && draw_method==2 && mask_array[j*input_img.height() + i]!=0 ||
                        draw_method==3 && seed_array[j*input_img.height() + i]!=0){
                    result_img.setPixelColor(j, i, result_array[j*input_img.height() + i]);

                }
            }
            else
                result_img.setPixelColor(j, i, Qt::transparent);
        }
    }

    // 绘制图像
    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    result_pix.fill(Qt::transparent);
    result_pix = QPixmap::fromImage( result_img.scaled(qSize, Qt::KeepAspectRatio) );
    update();

    // 内存回收
    delete []result_array;
    delete []seed_array;
}


// 锐化
void MainWindow::on_pushButton_6_clicked()
{
    process_pic(imgSharpen);
}

// 模糊
void MainWindow::on_pushButton_7_clicked()
{
    process_pic(imgBlur);

}

// 风格化
void MainWindow::on_pushButton_9_clicked()
{
    process_pic(imgCLAHE);
}

// 全局直方图均衡
void MainWindow::on_pushButton_8_clicked()
{
    process_pic(imgCLAHE_Global);
}

// 暂存图片：将结果区的图片保存到操作区，进行后续处理
// 即分别将input_pix, input_img, input_array的内容更新为结果区内容
void MainWindow::on_pushButton_10_clicked()
{
    input_pix = result_pix.copy();
    input_img = input_pix.scaled(originSize,Qt::KeepAspectRatio).toImage();;

    if (input_array != nullptr)
        delete []input_array;
    input_array = new int[input_img.height() * input_img.width()];
    for (int i=0; i<input_img.height(); i++){
        for (int j=0; j<input_img.width(); j++){
            input_array[j*input_img.height() + i] = (input_img.pixelColor(j, i).red()<<16)
                    + (input_img.pixelColor(j, i).green()<<8)
                    + (input_img.pixelColor(j, i).blue());

        }
    }
    update();
}

// 清空笔触：将画布填充为透明
void MainWindow::on_pushButton_11_clicked()
{
    pix.fill(Qt::transparent);
    update();
}


void MainWindow::on_comboBox_activated(int index)
{
    draw_method = index;
}
