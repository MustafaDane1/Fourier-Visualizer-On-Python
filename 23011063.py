import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGridLayout, QGroupBox, QSplitter, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt

class SinyalCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100, num_subplots=4, layout_rows=4, layout_cols=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(SinyalCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.axes = []
        for i in range(1, num_subplots + 1):
            self.axes.append(self.fig.add_subplot(layout_rows, layout_cols, i))
        if num_subplots > 0:
            try:
                initial_h_pad = 1.2 if num_subplots >= 7 else 1.8
                self.fig.tight_layout(pad=1.0, h_pad=initial_h_pad, w_pad=1.0)
            except Exception as e:

                print(f"SinyalCanvas __init__ içinde ilk tight_layout hatası: {e}. subplots_adjust deneniyor.")
                adjust_hspace = 0.5 if num_subplots >= 7 else 0.4 
                if num_subplots == 1: adjust_hspace = 0 
                try:
                    self.fig.subplots_adjust(hspace=adjust_hspace if num_subplots > 1 else None)
                except Exception as e2:
                    print(f"SinyalCanvas __init__ içinde subplots_adjust hatası: {e2}")

class SinyalIslemci(QMainWindow):
    def __init__(self):
        super().__init__()
        self.w0_value = 1.0  # Fourier w0 için başlangıç değeri
        self.T_value = 2 * np.pi / self.w0_value # Fourier T için hesaplanmış başlangıç değeri
        self.active_input = 'w0' # 'w0' veya 'T' olabilir, hangisinin en son kullanıcı tarafından düzenlendiğini belirtir
        self.setWindowTitle("Sinyal Analizi ve Sentezi")
        self.setGeometry(100, 100, 1300, 950)
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Sekme 1: Sinyal Sentezleyici
        self.synthesizer_tab = QWidget()
        self.tabs.addTab(self.synthesizer_tab, "Sinyal Sentezleyici")
        synthesizer_tab_layout = QHBoxLayout(self.synthesizer_tab)

        synthesizer_input_panel = QWidget()
        synthesizer_input_layout = QVBoxLayout(synthesizer_input_panel)

        for i in range(1, 4):
            group_box = QGroupBox(f"Sinyal {i}")
            grid = QGridLayout()
            grid.addWidget(QLabel("Genlik (Ak):"), 0, 0)
            amp_edit = QLineEdit("1.0")
            amp_edit.setObjectName(f"sentez_amp_{i}")
            grid.addWidget(amp_edit, 0, 1)

            grid.addWidget(QLabel("Frekans (fk - Hz):"), 1, 0)
            freq_edit = QLineEdit("1.0")
            freq_edit.setObjectName(f"sentez_freq_{i}")
            grid.addWidget(freq_edit, 1, 1)

            grid.addWidget(QLabel("Faz (θk - radyan):"), 2, 0)
            phase_edit = QLineEdit("0.0")
            phase_edit.setObjectName(f"sentez_phase_{i}")
            grid.addWidget(phase_edit, 2, 1)

            group_box.setLayout(grid)
            synthesizer_input_layout.addWidget(group_box)

        synthesizer_time_group = QGroupBox("Çizim Süresi")
        synthesizer_time_layout = QGridLayout()
        synthesizer_time_layout.addWidget(QLabel("Süre (s):"),0,0)
        self.sentez_time_edit = QLineEdit("2.0")
        self.sentez_time_edit.setObjectName("sentez_time")
        synthesizer_time_layout.addWidget(self.sentez_time_edit,0,1)
        synthesizer_time_group.setLayout(synthesizer_time_layout)
        synthesizer_input_layout.addWidget(synthesizer_time_group)

        draw_sentez_button = QPushButton("Sinyalleri Sentezle ve Çiz")
        draw_sentez_button.clicked.connect(self.draw_sentez_signals)
        synthesizer_input_layout.addWidget(draw_sentez_button)
        synthesizer_input_layout.addStretch()

        synthesizer_graph_panel = QWidget()
        synthesizer_graph_layout = QVBoxLayout(synthesizer_graph_panel)
        synthesizer_label = QLabel("Sentezlenmiş Sinyal Grafikleri")
        synthesizer_label.setAlignment(Qt.AlignCenter)
        synthesizer_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        synthesizer_graph_layout.addWidget(synthesizer_label)
        self.synthesizer_canvas = SinyalCanvas(self, width=10, height=16, num_subplots=7, layout_rows=7, layout_cols=1)
        synthesizer_graph_layout.addWidget(self.synthesizer_canvas, 1)

        synthesizer_splitter = QSplitter(Qt.Horizontal)
        synthesizer_splitter.addWidget(synthesizer_input_panel)
        synthesizer_splitter.addWidget(synthesizer_graph_panel)
        synthesizer_splitter.setSizes([250, 1050])
        synthesizer_tab_layout.addWidget(synthesizer_splitter)

        # Sekme 2: Fourier Serileri Analizi
        self.fourier_tab = QWidget()
        self.tabs.addTab(self.fourier_tab, "Fourier Serileri Analizi")
        fourier_tab_layout = QHBoxLayout(self.fourier_tab)

        fourier_input_panel = QWidget()
        fourier_input_layout = QVBoxLayout(fourier_input_panel)

        fourier_params_group = QGroupBox("Fourier Serisi Parametreleri")
        fourier_params_grid_layout = QGridLayout()
        fourier_params_grid_layout.addWidget(QLabel("a0 katsayısı:"), 0, 0)
        a0_edit = QLineEdit("0.0") # Varsayılan değer 0.0 olarak ayarlandı
        a0_edit.setObjectName("fourier_a0")
        fourier_params_grid_layout.addWidget(a0_edit, 0, 1)
        fourier_params_grid_layout.addWidget(QLabel("w0 (temel frekans):"), 1, 0)
        self.fourier_w0_edit = QLineEdit(f"{self.w0_value:.4f}")
        self.fourier_w0_edit.setObjectName("fourier_w0")
        fourier_params_grid_layout.addWidget(self.fourier_w0_edit, 1, 1)
        fourier_params_grid_layout.addWidget(QLabel("T (periyot):"), 2, 0)
        self.fourier_t_edit = QLineEdit(f"{self.T_value:.4f}")
        self.fourier_t_edit.setObjectName("fourier_T")
        fourier_params_grid_layout.addWidget(self.fourier_t_edit, 2, 1)
        fourier_params_group.setLayout(fourier_params_grid_layout)
        fourier_input_layout.addWidget(fourier_params_group)

        for i in range(1, 4):
            group_box = QGroupBox(f"k={i} Harmonik")
            grid = QGridLayout()
            grid.addWidget(QLabel(f"ak (a{i}):"), 0, 0)
            ak_edit = QLineEdit("0.0") # Varsayılan değer 0.0
            ak_edit.setObjectName(f"fourier_ak_{i}")
            grid.addWidget(ak_edit, 0, 1)
            grid.addWidget(QLabel(f"bk (b{i}):"), 1, 0)
            bk_edit = QLineEdit("0.0")
            bk_edit.setObjectName(f"fourier_bk_{i}")
            grid.addWidget(bk_edit, 1, 1)
            group_box.setLayout(grid)
            fourier_input_layout.addWidget(group_box)

        draw_fourier_button = QPushButton("Fourier Serisi Çiz")
        draw_fourier_button.clicked.connect(self.draw_fourier_signals)
        fourier_input_layout.addWidget(draw_fourier_button)
        fourier_input_layout.addStretch()

        fourier_graph_panel = QWidget()
        fourier_graph_layout = QVBoxLayout(fourier_graph_panel)
        fourier_label = QLabel("Fourier Serisi Grafikleri")
        fourier_label.setAlignment(Qt.AlignCenter)
        fourier_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        fourier_graph_layout.addWidget(fourier_label)
        self.fourier_canvas = SinyalCanvas(self, width=8, height=12, num_subplots=4, layout_rows=4, layout_cols=1)
        fourier_graph_layout.addWidget(self.fourier_canvas, 1)

        fourier_splitter = QSplitter(Qt.Horizontal)
        fourier_splitter.addWidget(fourier_input_panel)
        fourier_splitter.addWidget(fourier_graph_panel)
        fourier_splitter.setSizes([300, 1000])
        fourier_tab_layout.addWidget(fourier_splitter)

        # w0 ve T editörleri için sinyal bağlantıları
        self.fourier_w0_edit.editingFinished.connect(self.handle_fourier_w0_edited)
        self.fourier_t_edit.editingFinished.connect(self.handle_fourier_T_edited)

    def handle_fourier_w0_edited(self):
        try:
            new_w0_text = self.fourier_w0_edit.text()
            new_w0_val = float(new_w0_text)

            if not (new_w0_val > 0):
                QMessageBox.warning(self, "Geçersiz Değer", "w0 değeri pozitif bir sayı olmalıdır.")
                self.fourier_w0_edit.setText(f"{self.w0_value:.4f}") # Eski geçerli değere dön
                return

            # Eğer T aktif giriş ise ve kullanıcı w0'ı değiştirdiyse uyar
            if self.active_input == 'T' and abs(new_w0_val - self.w0_value) > 1e-6:
                reply = QMessageBox.question(self, "Değişiklik Onayı",
                                             f"w0 değeri, T ({self.T_value:.2f}) üzerinden {self.w0_value:.4f} olarak hesaplanmıştı.\n"
                                             f"Bu değeri {new_w0_text} olarak değiştirmek T değerini geçersiz kılacak ve w0 ana girdi olacaktır.\n"
                                             f"Devam etmek istiyor musunuz?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.fourier_w0_edit.setText(f"{self.w0_value:.4f}") # Değişikliği geri al
                    return
            
            self.w0_value = new_w0_val
            self.T_value = 2 * np.pi / self.w0_value
            self.fourier_t_edit.setText(f"{self.T_value:.4f}")
            self.active_input = 'w0'

        except ValueError:
            QMessageBox.warning(self, "Geçersiz Giriş", "w0 için sayısal bir değer girin.")
            self.fourier_w0_edit.setText(f"{self.w0_value:.4f}") # Eski geçerli değere dön

    def handle_fourier_T_edited(self):
        try:
            new_T_text = self.fourier_t_edit.text()
            new_T_val = float(new_T_text)

            if not (new_T_val > 0):
                QMessageBox.warning(self, "Geçersiz Değer", "T değeri pozitif bir sayı olmalıdır.")
                self.fourier_t_edit.setText(f"{self.T_value:.4f}") # Eski geçerli değere dön
                return

            # Eğer w0 aktif giriş ise ve kullanıcı T'yi değiştirdiyse uyar
            if self.active_input == 'w0' and abs(new_T_val - self.T_value) > 1e-6:
                reply = QMessageBox.question(self, "Değişiklik Onayı",
                                             f"T değeri, w0 ({self.w0_value:.2f}) üzerinden {self.T_value:.4f} olarak hesaplanmıştı.\n"
                                             f"Bu değeri {new_T_text} olarak değiştirmek w0 değerini geçersiz kılacak ve T ana girdi olacaktır.\n"
                                             f"Devam etmek istiyor musunuz?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.fourier_t_edit.setText(f"{self.T_value:.4f}") # Değişikliği geri al
                    return
            
            self.T_value = new_T_val
            self.w0_value = 2 * np.pi / self.T_value
            self.fourier_w0_edit.setText(f"{self.w0_value:.4f}")
            self.active_input = 'T'

        except ValueError:
            QMessageBox.warning(self, "Geçersiz Giriş", "T için sayısal bir değer girin.")
            self.fourier_t_edit.setText(f"{self.T_value:.4f}") # Eski geçerli değere dön

    def draw_fourier_signals(self):
        try:
            a0 = float(self.findChild(QLineEdit, "fourier_a0").text())
            # w0 ve T_period değerlerini self.w0_value ve self.T_value üzerinden al
            w0 = self.w0_value
            T_period = self.T_value

            if T_period <= 0: # Bu kontrol, değerlerin bir şekilde bozulması durumunda bir yedek olarak kalabilir
                if w0 > 0: T_period = 2 * np.pi / w0
                else: 
                    QMessageBox.warning(self, "Hesaplama Hatası", "Geçerli w0 veya T değeri olmadan çizim yapılamaz.")
                    return
            
            ak_values = []
            bk_values = []
            for i in range(1, 4):
                ak = float(self.findChild(QLineEdit, f"fourier_ak_{i}").text())
                bk = float(self.findChild(QLineEdit, f"fourier_bk_{i}").text())
                ak_values.append(ak)
                bk_values.append(bk)
            
            t = np.linspace(0, T_period, 500)
            
            components = []
            components.append(np.ones_like(t) * a0 / 2.0) 
            for i in range(3):
                k = i + 1
                components.append(ak_values[i] * np.cos(k * w0 * t))
                components.append(bk_values[i] * np.sin(k * w0 * t))
            
            fourier_series = np.sum(components, axis=0)
            self._draw_fourier_on_canvas(self.fourier_canvas, t, components, fourier_series, a0, ak_values, bk_values)
        except ValueError:
            print("Lütfen Fourier serisi için geçerli sayısal değerler girin.")
            return


    def _draw_fourier_on_canvas(self, canvas, t, components, fourier_series, a0_val, ak_vals, bk_vals):
        for ax in canvas.axes:
            ax.clear()
        
        colors_cos = ['green', 'purple', 'brown']
        colors_sin = ['lime', 'violet', 'sandybrown']
        
        title_fontsize = 10
        label_fontsize = 9
        tick_fontsize = label_fontsize -1

        canvas.axes[0].plot(t, components[0], color='blue', linewidth=1.5)
        canvas.axes[0].set_title(f"DC Bileşeni: a0/2 = {a0_val/2:.2f}", fontsize=title_fontsize)
        canvas.axes[0].grid(True, linestyle='--', alpha=0.7)
        canvas.axes[0].set_xlabel('Zaman (s)', fontsize=label_fontsize)
        canvas.axes[0].set_ylabel('Genlik', fontsize=label_fontsize)
        canvas.axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        ax_cos = canvas.axes[1]
        ax_cos.set_title("Kosinüs Terimleri (ak * cos(k*w0*t))", fontsize=title_fontsize)
        for i in range(3):
            k = i + 1
            ax_cos.plot(t, components[1 + 2*i], color=colors_cos[i], linewidth=1.5, label=f"a{k}*cos({k}w0t)")
        ax_cos.grid(True, linestyle='--', alpha=0.7)
        ax_cos.set_xlabel('Zaman (s)', fontsize=label_fontsize)
        ax_cos.set_ylabel('Genlik', fontsize=label_fontsize)
        if any(ak_vals): ax_cos.legend(fontsize=label_fontsize-1)
        ax_cos.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        ax_sin = canvas.axes[2]
        ax_sin.set_title("Sinüs Terimleri (bk * sin(k*w0*t))", fontsize=title_fontsize)
        for i in range(3):
            k = i + 1
            ax_sin.plot(t, components[2 + 2*i], color=colors_sin[i], linewidth=1.5, label=f"b{k}*sin({k}w0t)")
        ax_sin.grid(True, linestyle='--', alpha=0.7)
        ax_sin.set_xlabel('Zaman (s)', fontsize=label_fontsize)
        ax_sin.set_ylabel('Genlik', fontsize=label_fontsize)
        if any(bk_vals): ax_sin.legend(fontsize=label_fontsize-1)
        ax_sin.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        canvas.axes[3].plot(t, fourier_series, 'r', linewidth=2)
        canvas.axes[3].set_title("Toplam Fourier Serisi", fontsize=title_fontsize)
        canvas.axes[3].grid(True, linestyle='--', alpha=0.7)
        canvas.axes[3].set_xlabel('Zaman (s)', fontsize=label_fontsize)
        canvas.axes[3].set_ylabel('Genlik', fontsize=label_fontsize)
        canvas.axes[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        canvas.fig.tight_layout(pad=1.0, h_pad=1.8, w_pad=1.5)
        canvas.draw()

    def draw_sentez_signals(self):
        try:
            amplitudes = []
            frequencies = []
            phases = []
            for i in range(1, 4):
                amp = float(self.findChild(QLineEdit, f"sentez_amp_{i}").text())
                freq = float(self.findChild(QLineEdit, f"sentez_freq_{i}").text())
                phase = float(self.findChild(QLineEdit, f"sentez_phase_{i}").text())
                amplitudes.append(amp)
                frequencies.append(freq)
                phases.append(phase)
            
            plot_time = float(self.sentez_time_edit.text())
            if plot_time <= 0: plot_time = 2.0
            t = np.linspace(0, plot_time, 500)
            
            all_components = []
            for i in range(3):
                all_components.append(amplitudes[i] * np.cos(2 * np.pi * frequencies[i] * t + phases[i]))
            for i in range(3):
                all_components.append(amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i]))
            
            sum_signal = np.sum(all_components, axis=0)
            self._draw_sentez_on_canvas(self.synthesizer_canvas, t, all_components, sum_signal, amplitudes, frequencies, phases)
        except ValueError:
            print("Lütfen sentezleyici için geçerli sayısal değerler girin.")
            return

    def _draw_sentez_on_canvas(self, canvas, t, all_components, sum_signal, amplitudes, frequencies, phases):
        for ax in canvas.axes:
            ax.clear()

        cos_colors = ['blue', 'green', 'purple']
        sin_colors = ['cyan', 'lime', 'magenta']

        title_fontsize = 9
        label_fontsize = 8
        tick_fontsize = label_fontsize -1

        for i in range(3):
            canvas.axes[i].plot(t, all_components[i], color=cos_colors[i], linewidth=1.5)
            title_str = (f"Kosinüs {i+1}: A={amplitudes[i]}, f={frequencies[i]} Hz\n"
                         f"Faz={phases[i]:.2f} rad")
            canvas.axes[i].set_title(title_str, fontsize=title_fontsize)
            canvas.axes[i].grid(True, linestyle='--', alpha=0.7)
            canvas.axes[i].set_xlabel('Zaman (s)', fontsize=label_fontsize)
            canvas.axes[i].set_ylabel('Genlik', fontsize=label_fontsize)
            canvas.axes[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)


        for i in range(3):
            canvas.axes[i+3].plot(t, all_components[i+3], color=sin_colors[i], linewidth=1.5)
            title_str = (f"Sinüs {i+1}: A={amplitudes[i]}, f={frequencies[i]} Hz\n"
                         f"Faz={phases[i]:.2f} rad")
            canvas.axes[i+3].set_title(title_str, fontsize=title_fontsize)
            canvas.axes[i+3].grid(True, linestyle='--', alpha=0.7)
            canvas.axes[i+3].set_xlabel('Zaman (s)', fontsize=label_fontsize)
            canvas.axes[i+3].set_ylabel('Genlik', fontsize=label_fontsize)
            canvas.axes[i+3].tick_params(axis='both', which='major', labelsize=tick_fontsize)

        canvas.axes[6].plot(t, sum_signal, 'r', linewidth=2)
        canvas.axes[6].set_title("Toplam Sentez Sinyal", fontsize=title_fontsize+1)
        canvas.axes[6].grid(True, linestyle='--', alpha=0.7)
        canvas.axes[6].set_xlabel('Zaman (s)', fontsize=label_fontsize)
        canvas.axes[6].set_ylabel('Genlik', fontsize=label_fontsize)
        canvas.axes[6].tick_params(axis='both', which='major', labelsize=tick_fontsize)

        canvas.fig.tight_layout(pad=1.0, h_pad=1.2, w_pad=1.0)
        canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SinyalIslemci()
    window.show()
    sys.exit(app.exec_())