#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 10:38
# @Author  : CaoQixuan
# @File    : 废弃代码.py
# @Description :
# import sys
#
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog
# from PyQt5.QtGui import QPixmap
#
# class MyApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         # 创建文本框、按钮和标签
#         self.textbox = QLineEdit(self)
#         self.button = QPushButton('展示图片', self)
#         self.label = QLabel(self)
#
#         # 设置文本框和按钮的布局
#         hbox = QHBoxLayout()
#         hbox.addWidget(self.textbox)
#         hbox.addWidget(self.button)
#
#         # 设置主窗口的布局
#         vbox = QVBoxLayout()
#         vbox.addLayout(hbox)
#         vbox.addWidget(self.label)
#         self.setLayout(vbox)
#
#         # 连接按钮的点击事件
#         self.button.clicked.connect(self.showImage)
#
#         self.setWindowTitle('展示文本和图片')
#         self.show()
#
#     # 显示图片
#     def showImage(self):
#         fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files (*.jpg *.jpeg *.gif *.png *.bmp)')
#         if fname:
#             pixmap = QPixmap(fname)
#             self.label.setPixmap(pixmap)
#             self.label.setAlignment(Qt.AlignCenter)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = MyApp()
#     sys.exit(app.exec_())
#
# import sys
#
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPlainTextEdit, QPushButton, QFileDialog
#
# from PyQt.TimeWindow import TimePhase
#
#
# class MyWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("My Window")
#         self.resize(800, 600)
#         # 文本输入框
#         text_edit_label = QLabel("请输入文本：", self)
#         text_edit_label.setGeometry(20, 20, 100, 30)
#
#         self.text_edit = QPlainTextEdit(self)
#         self.text_edit.setGeometry(120, 20, 300, 150)
#
#         # 图片标签和选择按钮
#         img_label_label = QLabel("请选择图片：", self)
#         img_label_label.setGeometry(20, 190, 100, 30)
#
#         self.img_label = QLabel(self)
#         self.img_label.setGeometry(120, 190, 200, 200)
#
#         choose_btn = QPushButton("选择", self)
#         choose_btn.setGeometry(340, 265, 80, 30)
#         choose_btn.clicked.connect(self.choose_img)
#
#         # 提交按钮
#         submit_btn = QPushButton("提交", self)
#         submit_btn.setGeometry(450, 370, 100, 30)
#         submit_btn.clicked.connect(self.submit)
#
#         # 在新窗口中展示提交的文本和图片
#         self.new_window = QWidget()
#         self.new_window.setWindowTitle("New Window")
#         self.new_window.setGeometry(200, 200, 600, 600)
#         self.time = TimePhase()
#         self.time.closeSignal.connect(self.activeExit)
#
#     def choose_img(self):
#         # 选择图片文件
#         file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg)")
#         if file_name:
#             # 如果选择了图片文件则预览图片
#             self.img_label.setPixmap(QPixmap(file_name).scaledToWidth(200))
#
#     def submit(self):
#         # 获取输入的文本和图片
#         self.show()
#         text = self.text_edit.toPlainText()
#         # img_pixmap = self.imageShow.pixmap()
#
#         # 展示文本
#         text_label = QLabel(self.new_window)
#         text_label.setGeometry(100, 20, 500, 150)
#         text_label.setWordWrap(True)
#         text_label.setText(text)
#
#         # 展示图片
#         # imageShow = QLabel(self.disPlayWindow)
#         # imageShow.setGeometry(300, 300, 200, 200)
#         # imageShow.setPixmap(img_pixmap.scaledToWidth(200))
#
#         # 展示"Yes"字符
#         yes_btn = QPushButton("Yes", self.new_window)
#         yes_btn.setGeometry(250, 300, 60, 30)
#         # yes_btn.setStyleSheet("background-color: green; color: white;")
#         self.time.show()
#         self.time.myTimerState()
#         self.close()
#
#     def activeExit(self, info):
#         self.new_window.show()
#         print(info)
#
#     def keyPressEvent(self, event):
#         # 如果按下回车键，自动触发提交按钮
#         if event.key() == Qt.Key_Return:
#             self.submit()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MyWindow()
#     window.show()
#     app.exec_()
