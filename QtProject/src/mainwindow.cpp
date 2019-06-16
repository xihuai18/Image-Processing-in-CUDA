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




MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    isDrawing = false;
    lastPoint.setX(-INIT_WIDTH-BOUNDING);
    lastPoint.setY(-INIT_WIDTH-BOUNDING);
    endPoint.setX(-INIT_WIDTH-BOUNDING);
    endPoint.setY(-INIT_WIDTH-BOUNDING);
    penWidth = INIT_WIDTH;
    penColor = QColor DEL_COLOR;
    draw_method = 0;
}

MainWindow::~MainWindow()
{
    delete ui;
    delete []input_array;
}

void MainWindow::paintEvent(QPaintEvent *)
{
//      qDebug() << lastPoint << endPoint;
      QPen pen;
      pen.setWidth(penWidth);
      pen.setColor(penColor);
      QPainter painter(this);
      painter.drawPixmap(BOUNDING, BOUNDING, input_pix);
      painter.drawPixmap(2*BOUNDING+WINDOW_SIZE,BOUNDING, result_pix);
      if(isDrawing) //如果正在绘图，就在辅助画布上绘制
      {
         //将以前pix中的内容复制到tempPix中，保证以前的内容不消失
         QPainter pp(&pix);
         pp.setPen(pen);
         pp.drawLine(lastPoint, endPoint);    // 让前一个坐标值等于后一个坐标值，这样就能实现画出连续的线
         painter.drawPixmap(BOUNDING, BOUNDING, pix);
         lastPoint = endPoint;
      }
      else{
          painter.drawPixmap(BOUNDING, BOUNDING, pix);
      }

}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
               if(event->button()==Qt::LeftButton) //鼠标左键按下
               {
                   lastPoint = event->pos();
                   endPoint = event->pos();
//                   qDebug() << "press, last = " << lastPoint;
                   isDrawing = true;   //正在绘图

               }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{

         if(event->buttons()&Qt::LeftButton) //鼠标左键按下的同时移动鼠标
        {
                  endPoint = event->pos();
                  update(); //进行绘制
         }
//         qDebug() << "move, end = " << endPoint;
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{

            if(event->button() == Qt::LeftButton) //鼠标左键释放
             {
                      endPoint = event->pos();
                      update();
                      isDrawing = false;    //结束绘图
             }
//            qDebug() << "release, end = " << endPoint;
}


void MainWindow::on_pushButton_clicked()
{
    penColor = QColor DEL_COLOR;
}

void MainWindow::on_pushButton_2_clicked()
{
    penColor = QColor ADD_COLOR;
}

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    penWidth = value;
}

void MainWindow::on_pushButton_3_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Open File"), ".", tr("Images (*.png *.bmp *.jpg)"));
    // 暂时用来保存种子
    QPixmap save_pix = result_pix.scaled(originSize);
    bool flag = save_pix.save(fileName);
    qDebug()<<"SAVE"<<flag<<"SAVE WHERE"<<fileName;
}

// Load picture
void MainWindow::on_pushButton_4_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), ".", tr("Images (*.png *.bmp *.jpg)"));
    QPixmap tmp;
    bool flag = tmp.load(fileName);
    originSize = tmp.size();
    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    input_pix = tmp.scaled(qSize,Qt::KeepAspectRatio);
    qDebug()<<fileName<<flag<<qSize<<tmp.size()<<input_pix.size();

    pix = QPixmap(input_pix.size());
    pix.fill(Qt::transparent);//用透明色填充
    result_pix = input_pix.copy();

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

void MainWindow::on_pushButton_5_clicked()
{

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
    // 导出数据
    std::ofstream OpenFile("pic_array.txt");
    OpenFile<<seed_img.width()<<std::endl<<seed_img.height()<<std::endl;
    for (int i=0; i<seed_img.height()*seed_img.width(); i++){
        OpenFile<<input_array[i]<<" ";
    }
    for (int i=0; i<seed_img.height()*seed_img.width(); i++){
        OpenFile<<seed_array[i]<<" ";
    }
    OpenFile<<std::endl;
    OpenFile.close();

    std::clock_t startTime,endTime;
    startTime = std::clock();

    if (mask_array != nullptr)
        delete []mask_array;
    if (isGPU)
        mask_array = getCutMask(input_array, seed_array, seed_img.width(), seed_img.height());
    else
        mask_array = getCutMask_iSAP(input_array, seed_array, seed_img.width(), seed_img.height());

    endTime = std::clock();//计时结束

    qDebug()<<"CUT FINISH!"<< "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" ;
    QImage result_img = input_img.copy();
    result_img = result_img.convertToFormat(QImage::Format_ARGB32);
    for (int j=0; j<seed_img.width(); j++){
        for (int i=0; i<seed_img.height(); i++){
            if (mask_array[j*seed_img.height() + i] == 255){
                result_img.setPixelColor(j, i,  Qt::transparent);
            }
//            qDebug()<<j<<i<<mask_array[j*seed_img.height() + i] ;
        }
    }
    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    result_pix.fill(Qt::transparent);
    result_pix = QPixmap::fromImage( result_img.scaled(qSize, Qt::KeepAspectRatio) );
    update();
    delete []seed_array;
}

void MainWindow::on_checkBox_stateChanged(int arg1)
{
    qDebug()<<"is GPU? "<<arg1;
    if (arg1 == 0){
        isGPU = false;
    }
    else{
        isGPU = true;
    }
}

void MainWindow::process_pic(int* (*pf)(int*,int, int)){
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



    std::clock_t startTime,endTime;
    startTime = std::clock();
    int * result_array = pf(input_array, input_img.width(), input_img.height());
    endTime = std::clock();//计时结束
    qDebug()<<"FINISH!"<< "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" ;
    QImage result_img = input_img.copy();


    for (int j=0; j<input_img.width(); j++){
        for (int i=0; i<input_img.height(); i++){
            if (input_img.pixelColor(j, i) != Qt::transparent){
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

    QSize qSize(WINDOW_SIZE, WINDOW_SIZE);
    result_pix.fill(Qt::transparent);
    result_pix = QPixmap::fromImage( result_img.scaled(qSize, Qt::KeepAspectRatio) );
    update();
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

// 暂存图片
void MainWindow::on_pushButton_10_clicked()
{
    input_pix = result_pix.copy();
    qDebug()<<input_pix.size();
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

// 清空笔触
void MainWindow::on_pushButton_11_clicked()
{
    pix.fill(Qt::transparent);
    update();
}


void MainWindow::on_comboBox_activated(int index)
{
    draw_method = index;
}
