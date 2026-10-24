"""
Сейсмические атрибуты для куба амплитуд shape (X, Y, T).

Атрибуты
--------
Геометрические (требуют сектор с соседними трассами):
    coherence, dip, azimuth, chaos
    similarity
    k_mean, k_gauss, k1, k2, k_max, k_min,
    k_pos, k_neg, k_dip, k_strike, curvedness, shape_index

Трассовые (независимые, достаточно одной трассы):
    inst_amplitude, inst_phase, inst_frequency, inst_bandwidth,
    reflection_strength_gradient, sweetness
    peak_frequency, peak_magnitude, cwt_bandwidth,
    total_energy_cwt, magnitude_at_{f}hz, q_factor_{dt}ms
    rms_amplitude, total_energy, rai
    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_entropy

Система координат
-----------------
    x — инлайн    (ось 0)
    y — кросслайн (ось 1)
    t — время     (ось 2)

Использование
-------------
    from attributes import SeismicAttributes

    # Для сектора (старый режим — полная совместимость)
    sa = SeismicAttributes(sector, sigma_deriv=1.0, sigma_smooth=2.0)
    attrs = sa.compute(['coherence', 'similarity', 'inst_bandwidth'])
    # → dict[str, np.ndarray], каждый shape (X, Y, T)

    # Или отдельные атрибуты напрямую
    coh = sa.coherence()   # np.ndarray (X, Y, T)
    sim = sa.similarity()  # np.ndarray (X, Y, T)

    # Все атрибуты (обратная совместимость с dataset.ipynb)
    attrs = sa.compute_all(dt_ms=2.0)
"""

__version__ = '2.0.0'
__all__     = ['SeismicAttributes', 'AVAILABLE_ATTRS']

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import hilbert
import pywt
from joblib import Parallel, delayed


# Полный список доступных непараметрических атрибутов.
# Параметрические (magnitude_at_{f}hz, q_factor_{dt}ms) сюда не входят —
# они обрабатываются отдельно в compute() по префиксу имени.
AVAILABLE_ATTRS = [
    # GST
    'coherence', 'dip', 'azimuth', 'chaos',
    # Similarity
    'similarity',
    # Curvature
    'k_mean', 'k_gauss', 'k1', 'k2', 'k_max', 'k_min',
    'k_pos', 'k_neg', 'k_dip', 'k_strike', 'curvedness', 'shape_index',
    # Instant
    'inst_amplitude', 'inst_phase', 'inst_frequency', 'inst_bandwidth',
    'reflection_strength_gradient', 'sweetness',
    # Spectral CWT
    'peak_frequency', 'peak_magnitude', 'cwt_bandwidth', 'total_energy_cwt',
    # Amplitude
    'rms_amplitude', 'total_energy', 'rai',
    # Texture GLCM
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_entropy',
]

# ── Вспомогательные функции уровня модуля ─────────────────────────────────

def _cwt_one_trace(tr, scales, wav, fs):
    """
    CWT одной трассы вейвлетом Морле.

    tr     : np.ndarray, shape (T,) — одна трасса амплитуд
    scales : np.ndarray, shape (n_freqs,) — масштабы вейвлета (из freq через central_frequency)
    wav    : str — имя вейвлета (например 'cmor1.5-1.0')
    fs     : float — частота дискретизации [Гц]

    Возвращает np.ndarray, shape (n_freqs, T), float32 — модуль CWT-коэффициентов,
    безразмерная амплитуда ≥ 0.
    """
    coeffs, _ = pywt.cwt(tr, scales, wav, sampling_period=1.0/fs)
    return np.abs(coeffs).astype(np.float32)

# ── Основной класс ─────────────────────────────────────────────────────────

class SeismicAttributes:
    """
    Вычисляет сейсмические атрибуты для куба амплитуд.

    Промежуточные вычисления (eigendecomposition GST, наклоны p/q,
    преобразование Гильберта, CWT) кэшируются — при запросе нескольких
    атрибутов одной группы пересчёта не происходит.

    cube         : np.ndarray, shape (X, Y, T) — куб амплитуд (сектор или полный куб)
    sigma_deriv  : float — σ дифференцирующего фильтра Гаусса [сэмплы], по умолчанию 1.0
    sigma_smooth : float — σ сглаживающего окна GST [сэмплы], по умолчанию 2.0
    dt_ms        : float — шаг дискретизации [мс], нужен для inst/spectral атрибутов, по умолчанию 2.0

    Возвращает экземпляр SeismicAttributes с self.shape = cube.shape (X, Y, T).
    """

    def __init__(self, cube: np.ndarray,
                 sigma_deriv: float = 1.0,
                 sigma_smooth: float = 2.0,
                 dt_ms: float = 2.0):
        self.cube         = cube.astype(np.float32)
        self.sigma_deriv  = sigma_deriv
        self.sigma_smooth = sigma_smooth
        self.dt_ms        = dt_ms
        self.shape        = cube.shape   # (X, Y, T)

        # ── Кэш GST eigendecomposition ────────────────────────────────────────
        self._nx   = None
        self._ny   = None
        self._nt   = None
        self._lam1 = None
        self._lam2 = None
        self._lam3 = None

        # ── Кэш наклонов ──────────────────────────────────────────────────────
        self._p = None
        self._q = None

        # ── Кэш преобразования Гильберта ──────────────────────────────────────
        self._analytic        = None
        self._inst_amplitude  = None
        self._inst_phase      = None
        self._inst_frequency  = None

        # ── Кэш CWT ───────────────────────────────────────────────────────────
        self._cwt_mag   = None   # (X*Y, n_freqs, T)
        self._cwt_freqs = None   # (n_freqs,)
        self._cwt_params = None  # (dt_ms, freq_min, freq_max, n_freqs) — для инвалидации

        self._glcm_cache  = None   # dict со статистиками
        self._glcm_params = None   # (axes, window, offset, n_levels) для инвалидации

    # =========================================================================
    # Внутренние методы: вычисление и кэширование промежуточных результатов
    # =========================================================================

    def _gaussian_gradient(self, axis: int) -> np.ndarray:
        """
        Производная куба по заданной оси через дифференцирующий фильтр Гаусса.

        axis : int — ось дифференцирования (0=X/инлайн, 1=Y/кросслайн, 2=T/время)

        Возвращает np.ndarray, shape (X, Y, T), float32 — производная куба вдоль
        `axis`, сглаженная по σ=self.sigma_deriv [сэмплы].
        """
        order       = [0, 0, 0]
        order[axis] = 1
        return gaussian_filter(self.cube, sigma=self.sigma_deriv, order=order)

    def _build_gst(self) -> tuple:
        """
        Компоненты тензора структурных градиентов (GST) — попарные произведения
        градиентов gx,gy,gt, сглаженные окном σ=self.sigma_smooth.

        Входных параметров нет — использует self.cube через _gaussian_gradient().

        Возвращает tuple из 6 np.ndarray, каждый shape (X, Y, T), float32:
        (Txx, Tyy, Ttt, Txy, Txt, Tyt) — элементы симметричного 3×3 тензора в каждой точке.
        """
        gx = self._gaussian_gradient(axis=0)
        gy = self._gaussian_gradient(axis=1)
        gt = self._gaussian_gradient(axis=2)
        s  = lambda a: gaussian_filter(a, sigma=self.sigma_smooth)
        return (s(gx*gx), s(gy*gy), s(gt*gt),
                s(gx*gy), s(gx*gt), s(gy*gt))

    def _eigendecompose(self):
        """
        GST eigendecomposition — собственные значения/векторы локального тензора
        структуры в каждой точке куба. Вычисляется один раз, результат кэшируется.

        Входных параметров нет — использует self.cube через _build_gst().

        Возвращает None. Сохраняет в self._lam1 ≥ self._lam2 ≥ self._lam3
        (собственные значения, shape (N,), N=X*Y*T, безразмерные, энергия локальной
        структуры по убыванию) и self._nx/_ny/_nt (компоненты старшего собственного
        вектора, shape (N,), безразмерные, |v|=1).
        """
        if self._nx is not None:
            return
        Txx, Tyy, Ttt, Txy, Txt, Tyt = self._build_gst()
        T_flat = np.stack([
            np.stack([Txx.ravel(), Txy.ravel(), Txt.ravel()], axis=1),
            np.stack([Txy.ravel(), Tyy.ravel(), Tyt.ravel()], axis=1),
            np.stack([Txt.ravel(), Tyt.ravel(), Ttt.ravel()], axis=1),
        ], axis=1)                                    # (N, 3, 3)
        eigvals, eigvecs = np.linalg.eigh(T_flat)    # (N,3), (N,3,3)
        self._lam1 = eigvals[:, 2]
        self._lam2 = eigvals[:, 1]
        self._lam3 = eigvals[:, 0]
        v1         = eigvecs[:, :, 2]
        self._nx   = v1[:, 0]
        self._ny   = v1[:, 1]
        self._nt   = v1[:, 2]

    def _get_pq(self):
        """
        Локальные наклоны поверхности p=dt/dx, q=dt/dy — из старшего собственного
        вектора GST. База для всех атрибутов кривизны.

        Входных параметров нет — использует self._nx/_ny/_nt (через _eigendecompose()).

        Возвращает tuple (p, q), каждый np.ndarray shape (N,) float32, N=X*Y*T —
        безразмерные локальные наклоны отражателя вдоль X и Y соответственно.
        """
        if self._p is not None:
            return self._p, self._q
        self._eigendecompose()
        safe_nt  = np.where(np.abs(self._nt) > 1e-6, self._nt, 1e-6)
        self._p  = (-self._nx / safe_nt).astype(np.float32)
        self._q  = (-self._ny / safe_nt).astype(np.float32)
        return self._p, self._q

    def _ensure_hilbert(self):
        """
        Преобразование Гильберта по времени + мгновенные амплитуда/фаза/частота.
        Вычисляется один раз, результат кэшируется.

        Входных параметров нет — использует self.cube и self.dt_ms.

        Возвращает None. Сохраняет self._inst_amplitude [safe, ≥0], self._inst_phase
        [рад, -π..π], self._inst_frequency [Гц] — каждый np.ndarray shape (X, Y, T), float32.
        """
        if self._analytic is not None:
            return
        dt = self.dt_ms / 1000.0
        self._analytic       = hilbert(self.cube, axis=2)
        self._inst_amplitude = np.abs(self._analytic).astype(np.float32)
        self._inst_phase     = np.angle(self._analytic).astype(np.float32)
        freq = np.diff(np.unwrap(np.angle(self._analytic), axis=2), axis=2) / (2 * np.pi * dt)
        self._inst_frequency = np.concatenate(
            [freq, freq[:, :, -1:]], axis=2).astype(np.float32)

    def _ensure_cwt(self, freq_min=5.0, freq_max=120.0, n_freqs=64):
        """
        Непрерывное вейвлет-преобразование (CWT, вейвлет Морле) каждой трассы куба.
        Кэшируется по (dt_ms, freq_min, freq_max, n_freqs) — повторный вызов с теми
        же параметрами не пересчитывает.

        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки (линейно от freq_min до freq_max)

        Возвращает None. Сохраняет self._cwt_mag (shape (X*Y, n_freqs, T), float32,
        амплитуда CWT, безразмерная, ≥0) и self._cwt_freqs (shape (n_freqs,), [Гц]).
        """
        params = (self.dt_ms, freq_min, freq_max, n_freqs)
        if self._cwt_params == params:
            return
        X, Y, T = self.shape
        fs      = 1000.0 / self.dt_ms
        freqs   = np.linspace(freq_min, freq_max, n_freqs)
        wav     = 'cmor1.5-1.0'
        fc      = pywt.central_frequency(wav)
        scales  = (fc * fs) / freqs
        traces  = self.cube.reshape(-1, T)

        results = Parallel(n_jobs=-1)(
            delayed(_cwt_one_trace)(tr, scales, wav, fs) for tr in traces
        )
        self._cwt_mag    = np.stack(results, axis=0)  # (X*Y, n_freqs, T)
        self._cwt_freqs  = freqs
        self._cwt_params = params

    def _glcm_stats(self, axes: tuple = (0, 2), window: int = 5,
                offset: int = 1, n_levels: int = 8) -> dict:
        """
        Локальные Haralick-текстуры (GLCM) в плоскости, заданной `axes`.

        axes     : (0,2)=X-T (вертикальный разрез), (1,2)=Y-T, (0,1)=X-Y (карта)
        window   : int — размер окна вдоль каждой из двух осей плоскости [сэмплы]
        offset   : int — смещение для пары co-occurrence, вдоль axes[1] [сэмплы]
        n_levels : int — число уровней квантования амплитуды (локально, по перцентилям
                   сектора — устойчиво к разнице масштаба амплитуд между скважинами)

        Возвращает dict с 'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
        'glcm_entropy', каждый np.ndarray shape (X, Y, T), float32, безразмерные.
        """
        ax1, ax2 = axes
        cube = self.cube

        # Квантование — по перцентилям ЭТОГО сектора (не глобально!),
        # иначе текстура будет зависеть от абсолютного масштаба амплитуд
        # конкретной скважины/зоны, а не от формы паттерна.
        lo, hi = np.percentile(cube, [1, 99])
        scale  = max(hi - lo, 1e-8)
        q = np.clip(((cube - lo) / scale) * n_levels, 0, n_levels - 1 - 1e-6).astype(np.int32)

        # co-occurrence пара: q(x) против q(x+offset) вдоль axes[1]
        q_shift = np.roll(q, -offset, axis=ax2)

        win = [1, 1, 1]
        win[ax1] = window
        win[ax2] = window

        contrast    = np.zeros(self.shape, dtype=np.float32)
        homogeneity = np.zeros(self.shape, dtype=np.float32)
        energy      = np.zeros(self.shape, dtype=np.float32)
        entropy     = np.zeros(self.shape, dtype=np.float32)

        for i in range(n_levels):
            for j in range(n_levels):
                indicator = ((q == i) & (q_shift == j)).astype(np.float32)
                p_ij = uniform_filter(indicator, size=win)   # локальная p(i,j)

                diff2 = (i - j) ** 2
                contrast    += p_ij * diff2
                homogeneity += p_ij / (1.0 + diff2)
                energy      += p_ij ** 2
                entropy     += -p_ij * np.log(p_ij + 1e-8)

        return {
            'glcm_contrast': contrast,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_entropy': entropy,
        }

    def _ensure_glcm(self, axes=(0, 2), window=5, offset=1, n_levels=8):
        """
        Пересчитывает и кэширует 4 GLCM-статистики — только при смене параметров.

        axes, window, offset, n_levels — см. _glcm_stats().

        Возвращает None. Сохраняет в self._glcm_cache (dict, см. _glcm_stats())
        и self._glcm_params (кортеж параметров, для проверки инвалидации кэша).
        """
        params = (axes, window, offset, n_levels)
        if self._glcm_params == params:
            return
        self._glcm_cache  = self._glcm_stats(axes, window, offset, n_levels)
        self._glcm_params = params

    def _curvature_coeffs(self, sigma_curv: float = 1.0):
        """
        Коэффициенты локальной квадратичной аппроксимации поверхности отражателя,
        нужные для всех формул кривизны.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов p,q [сэмплы]

        Возвращает tuple (a, b, c, d, e), каждый np.ndarray shape (X, Y, T), float32 —
        коэффициенты локальной поверхности (a,b,c — вторые производные/полукривизны,
        d=p, e=q — сами наклоны), безразмерные.
        """
        p_vol, q_vol = self._get_pq()
        p = p_vol.reshape(self.shape)
        q = q_vol.reshape(self.shape)
        dp_dx = gaussian_filter(p, sigma=sigma_curv, order=[1, 0, 0])
        dp_dy = gaussian_filter(p, sigma=sigma_curv, order=[0, 1, 0])
        dq_dx = gaussian_filter(q, sigma=sigma_curv, order=[1, 0, 0])
        dq_dy = gaussian_filter(q, sigma=sigma_curv, order=[0, 1, 0])
        return (0.5*dp_dx, 0.5*dq_dy, 0.5*(dq_dx+dp_dy), p, q)

    def _k_mean_gauss(self, sigma_curv: float = 1.0):
        """
        Средняя и Гауссова кривизна поверхности — база для всех производных кривизн.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов p,q [сэмплы]

        Возвращает tuple (k_mean, k_gauss, a, b, c, d, e): k_mean и k_gauss —
        np.ndarray shape (X, Y, T), float32, безразмерные; a,b,c,d,e — см. _curvature_coeffs().
        """
        a, b, c, d, e = self._curvature_coeffs(sigma_curv)
        denom_32 = np.maximum(1.0 + d**2 + e**2, 1e-10) ** 1.5
        denom_2  = (1.0 + d**2 + e**2) ** 2
        k_mean  = (a*(1+e**2) + b*(1+d**2) - c*d*e) / denom_32
        k_gauss = (4*a*b - c**2) / denom_2
        return k_mean, k_gauss, a, b, c, d, e

    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        """
        Приводит плоский массив eigendecomposition к форме куба.

        arr : np.ndarray, shape (N,), N=X*Y*T — плоский массив (например self._lam1)

        Возвращает np.ndarray, shape (X, Y, T), float32.
        """
        return arr.reshape(self.shape).astype(np.float32)

    # =========================================================================
    # Публичные методы: отдельные атрибуты
    # =========================================================================

    # ── GST ───────────────────────────────────────────────────────────────────

    def coherence(self) -> np.ndarray:
        """
        C3-когерентность из собственных значений GST: (λ1−λ2)/(λ1+λ2) — планарность
        локальной структуры отражений.

        Входных параметров нет — использует self.cube (через _eigendecompose()).

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, диапазон [0, 1]:
        →1 гладкий протяжённый рефлектор, →0 разлом/хаос.
        """
        self._eigendecompose()
        d = self._lam1 + self._lam2
        return self._reshape(
            (self._lam1 - self._lam2) / np.where(d > 1e-10, d, 1e-10))

    def dip(self) -> np.ndarray:
        """
        Угол наклона отражателя (dip) от горизонтали, из старшего собственного
        вектора GST.

        Входных параметров нет — использует self.cube (через _eigendecompose()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [градусы], диапазон [0°, 90°]:
        0° — горизонтальный рефлектор, 90° — вертикальный.
        """
        self._eigendecompose()
        horiz = np.sqrt(self._nx**2 + self._ny**2)
        return self._reshape(
            np.degrees(np.arctan2(horiz, np.abs(self._nt) + 1e-10)))

    def azimuth(self) -> np.ndarray:
        """
        Азимут направления максимального падения отражателя, из проекции старшего
        собственного вектора GST на плоскость X-Y.

        Входных параметров нет — использует self.cube (через _eigendecompose()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [градусы], диапазон [0°, 360°).
        """
        self._eigendecompose()
        return self._reshape(
            np.degrees(np.arctan2(self._ny, self._nx)) % 360)

    def chaos(self) -> np.ndarray:
        """
        Мера хаотичности локальной структуры отражений: (2·λ2)/(λ1+λ3) − 1,
        из собственных значений GST.

        Входных параметров нет — использует self.cube (через _eigendecompose()).

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная,
        диапазон [-1, 1]: →1 хаотичная/разрушенная структура, →−1 планарный
        протяжённый рефлектор.
        """
        self._eigendecompose()
        d = self._lam1 + self._lam3
        return self._reshape(
            2.0 * self._lam2 / np.where(d > 1e-10, d, 1e-10) - 1.0)

    # ── Similarity ────────────────────────────────────────────────────────────

    def similarity(self, window_xy: int = 2, window_t: int = 5) -> np.ndarray:
        """
        C1-когерентность (Marfurt et al., 1998) — нормированное косинусное сходство
        соседних трасс в скользящем окне.

        window_xy : int — полуразмер окна по X и Y [трассы], итоговое окно (2·window_xy+1)
        window_t  : int — полуразмер окна по времени [сэмплы], итоговое окно (2·window_t+1)

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, диапазон [0, 1]:
        →1 идентичные соседние трассы, →0 разломы/хаос.
        """
        size = [2*window_xy+1, 2*window_xy+1, 2*window_t+1]
        N    = (2*window_xy + 1)**2
        mean_tr  = uniform_filter(self.cube,      size=size)
        mean_tr2 = uniform_filter(self.cube ** 2, size=size)
        denom    = mean_tr2 * N
        safe     = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
        return np.clip(mean_tr**2 / safe, 0.0, 1.0).astype(np.float32)

    # ── Curvature ─────────────────────────────────────────────────────────────

    def k_mean(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Средняя кривизна поверхности (k1+k2)/2, из локальных наклонов p,q.
        База для остальных кривизн.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл], без строгого
        ограничения диапазона: 0 — плоская поверхность, |k| растёт с изгибом.
        """
        km, _, *_ = self._k_mean_gauss(sigma_curv)
        return km.astype(np.float32)

    def k_gauss(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Гауссова кривизна поверхности k1·k2 — знак определяет тип формы.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл²]: >0 купол/чаша,
        <0 седло, 0 плоскость/цилиндр.
        """
        _, kg, *_ = self._k_mean_gauss(sigma_curv)
        return kg.astype(np.float32)

    def _principal_curvatures(self, sigma_curv: float = 1.0):
        """
        Главные кривизны k1, k2 — из (k_mean, k_gauss) через дискриминант.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает tuple (k1, k2), каждый np.ndarray shape (X, Y, T), float32,
        [1/сэмпл]: k1 — бОльшая по модулю главная кривизна, k2 — меньшая.
        """
        km, kg, *_ = self._k_mean_gauss(sigma_curv)
        disc = np.sqrt(np.maximum(km**2 - kg, 0.0))
        return km + disc, km - disc  # k1, k2

    def k1(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Первая (наибольшая по модулю) главная кривизна поверхности.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        k1_, _ = self._principal_curvatures(sigma_curv)
        return k1_.astype(np.float32)

    def k2(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Вторая (наименьшая по модулю) главная кривизна поверхности.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        _, k2_ = self._principal_curvatures(sigma_curv)
        return k2_.astype(np.float32)

    def k_max(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Кривизна, максимальная по модулю, из (k1, k2), со знаком — самый
        выраженный изгиб поверхности.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        k1_, k2_ = self._principal_curvatures(sigma_curv)
        return np.where(np.abs(k1_) >= np.abs(k2_), k1_, k2_).astype(np.float32)

    def k_min(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Кривизна, минимальная по модулю, из (k1, k2), со знаком.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        k1_, k2_ = self._principal_curvatures(sigma_curv)
        return np.where(np.abs(k1_) >= np.abs(k2_), k2_, k1_).astype(np.float32)

    def k_pos(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Положительная (антиклинальная/выпуклая) компонента кривизны:
        (a+b) + sqrt((a-b)²+c²).

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл], значения ≥
        соответствующей отрицательной компоненты (k_neg).
        """
        km, _, a, b, c, *_ = self._k_mean_gauss(sigma_curv)
        return ((a + b) + np.sqrt((a - b)**2 + c**2)).astype(np.float32)

    def k_neg(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Отрицательная (синклинальная/вогнутая) компонента кривизны:
        (a+b) − sqrt((a-b)²+c²).

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        km, _, a, b, c, *_ = self._k_mean_gauss(sigma_curv)
        return ((a + b) - np.sqrt((a - b)**2 + c**2)).astype(np.float32)

    def k_dip(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Кривизна вдоль направления максимального наклона (dip curvature) —
        компонента вдоль вектора (p,q).

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        km, _, a, b, c, d, e = self._k_mean_gauss(sigma_curv)
        d2e2     = d**2 + e**2
        safe_d2e2 = np.where(d2e2 > 1e-10, d2e2, 1e-10)
        denom_32  = np.maximum(1.0 + d2e2, 1e-10) ** 1.5
        return (2.0 * (a*d**2 + b*e**2 + c*d*e) /
                (safe_d2e2 * denom_32)).astype(np.float32)

    def k_strike(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Кривизна вдоль направления простирания (strike curvature) — компонента
        ортогонально dip.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл].
        """
        km, _, a, b, c, d, e = self._k_mean_gauss(sigma_curv)
        d2e2      = d**2 + e**2
        safe_d2e2 = np.where(d2e2 > 1e-10, d2e2, 1e-10)
        denom_sq  = np.sqrt(np.maximum(1.0 + d2e2, 1e-10))
        return (2.0 * (a*e**2 + b*d**2 - c*d*e) /
                (safe_d2e2 * denom_sq)).astype(np.float32)

    def curvedness(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Curvedness (Koenderink, 1990) — интенсивность искривления поверхности
        независимо от формы: sqrt((k1²+k2²)/2).

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, [1/сэмпл], диапазон ≥0:
        высокие значения — сильный локальный изгиб (потенциальные зоны
        трещиноватости/разлома).
        """
        k1_, k2_ = self._principal_curvatures(sigma_curv)
        kmax = np.where(np.abs(k1_) >= np.abs(k2_), k1_, k2_)
        kmin = np.where(np.abs(k1_) >= np.abs(k2_), k2_, k1_)
        return np.sqrt((kmax**2 + kmin**2) / 2.0).astype(np.float32)

    def shape_index(self, sigma_curv: float = 1.0) -> np.ndarray:
        """
        Shape Index (Koenderink, 1990) — форма поверхности независимо от
        интенсивности искривления.

        sigma_curv : float — σ фильтра Гаусса при дифференцировании наклонов [сэмплы]

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная,
        диапазон [-1, 1]: -1 чаша, 0 седло, +1 купол.
        """
        k1_, k2_ = self._principal_curvatures(sigma_curv)
        diff     = k1_ - k2_
        safe_diff = np.where(np.abs(diff) > 1e-10, diff, 1e-10)
        return ((2.0 / np.pi) * np.arctan((k1_ + k2_) / safe_diff)).astype(np.float32)

    # ── Мгновенные ────────────────────────────────────────────────────────────

    def inst_amplitude(self) -> np.ndarray:
        """
        Огибающая сигнала (instantaneous amplitude / reflection strength) —
        модуль аналитического сигнала.

        Входных параметров нет — использует self.cube (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, в единицах амплитуды
        куба, диапазон ≥0.
        """
        self._ensure_hilbert()
        return self._inst_amplitude

    def inst_phase(self) -> np.ndarray:
        """
        Мгновенная фаза сигнала (instantaneous phase).

        Входных параметров нет — использует self.cube (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [рад], диапазон [−π, π].
        """
        self._ensure_hilbert()
        return self._inst_phase

    def inst_frequency(self) -> np.ndarray:
        """
        Мгновенная частота сигнала (instantaneous frequency) — производная
        развёрнутой мгновенной фазы по времени.

        Входных параметров нет — использует self.cube и self.dt_ms (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [Гц].
        """
        self._ensure_hilbert()
        return self._inst_frequency

    def inst_bandwidth(self) -> np.ndarray:
        """
        Мгновенная полоса частот — производная огибающей, нормированная на саму
        огибающую (масштабно-инвариантная мера нестабильности сигнала).

        Входных параметров нет — использует self.cube и self.dt_ms (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [Гц].
        """
        self._ensure_hilbert()
        dA       = np.diff(self._inst_amplitude, axis=2)
        dA       = np.concatenate([dA, dA[:, :, -1:]], axis=2)
        safe_amp = np.where(self._inst_amplitude > 1e-10,
                            self._inst_amplitude, 1e-10)
        return (dA / (2 * np.pi * safe_amp)).astype(np.float32)

    def reflection_strength_gradient(self) -> np.ndarray:
        """
        dA/dt огибающей (reflection strength) без нормировки на амплитуду —
        в отличие от inst_bandwidth. Ловит резкие абсолютные контрасты
        отражательной способности (границы пласта/коллектора), а не
        относительную "нестабильность" сигнала.

        Входных параметров нет — использует self.cube и self.dt_ms (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, [единицы амплитуды/с],
        зависит от абсолютного масштаба амплитуд куба (см. предостережение об АРУ).
        """
        self._ensure_hilbert()
        dt = self.dt_ms / 1000.0
        dA = np.diff(self._inst_amplitude, axis=2) / dt
        dA = np.concatenate([dA, dA[:, :, -1:]], axis=2)
        return dA.astype(np.float32)

    def sweetness(self) -> np.ndarray:
        """
        Индикатор газа (sweetness) — inst_amplitude / sqrt(|inst_frequency|).

        Входных параметров нет — использует self.cube и self.dt_ms (через _ensure_hilbert()).

        Возвращает np.ndarray, shape (X, Y, T), float32, диапазон ≥0: высокие
        значения — сильная амплитуда на низкой частоте (классический индикатор газа).
        """
        self._ensure_hilbert()
        safe_freq = np.where(np.abs(self._inst_frequency) > 1e-10,
                             np.abs(self._inst_frequency), 1e-10)
        return (self._inst_amplitude / np.sqrt(safe_freq)).astype(np.float32)

    # ── Спектральные CWT ──────────────────────────────────────────────────────

    def peak_frequency(self, freq_min: float = 5.0, freq_max: float = 120.0,
                       n_freqs: int = 64) -> np.ndarray:
        """
        Частота с максимальной амплитудой локального CWT-спектра.

        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32, [Гц], диапазон
        [freq_min, freq_max].
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T = self.shape
        peak_idx = np.argmax(self._cwt_mag, axis=1)
        return self._cwt_freqs[peak_idx].reshape(X, Y, T).astype(np.float32)

    def peak_magnitude(self, freq_min: float = 5.0, freq_max: float = 120.0,
                       n_freqs: int = 64) -> np.ndarray:
        """
        Амплитуда CWT-спектра на частоте peak_frequency.

        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная амплитуда, ≥0.
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T  = self.shape
        peak_idx = np.argmax(self._cwt_mag, axis=1)
        return np.take_along_axis(
            self._cwt_mag, peak_idx[:, np.newaxis, :], axis=1
        ).squeeze(1).reshape(X, Y, T).astype(np.float32)

    def cwt_bandwidth(self, freq_min: float = 5.0, freq_max: float = 120.0,
                      n_freqs: int = 64) -> np.ndarray:
        """
        Ширина CWT-спектра на уровне −3 дБ (1/√2 от пиковой амплитуды).

        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32, [Гц], диапазон [0, freq_max-freq_min].
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T  = self.shape
        mag      = self._cwt_mag
        peak_idx = np.argmax(mag, axis=1)
        peak_mag = np.take_along_axis(
            mag, peak_idx[:, np.newaxis, :], axis=1).squeeze(1)
        threshold = peak_mag / np.sqrt(2)
        above     = mag >= threshold[:, np.newaxis, :]
        f_col     = self._cwt_freqs[np.newaxis, :, np.newaxis]
        f_masked  = np.where(above, np.broadcast_to(f_col, mag.shape), np.nan)
        bw_high   = np.nanmax(f_masked, axis=1)
        bw_low    = np.nanmin(f_masked, axis=1)
        bw        = np.where(np.isnan(bw_high) | np.isnan(bw_low),
                             0.0, bw_high - bw_low)
        return bw.reshape(X, Y, T).astype(np.float32)

    def total_energy_cwt(self, freq_min: float = 5.0, freq_max: float = 120.0,
                         n_freqs: int = 64) -> np.ndarray:
        """
        Суммарная спектральная мощность по всем частотам CWT.

        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32, диапазон ≥0.
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T = self.shape
        return (self._cwt_mag**2).sum(axis=1).reshape(X, Y, T).astype(np.float32)

    def magnitude_at(self, freq_hz: float,
                     freq_min: float = 5.0, freq_max: float = 120.0,
                     n_freqs: int = 64) -> np.ndarray:
        """
        Амплитуда CWT-спектра на заданной частоте (ближайшая точка сетки).

        freq_hz  : float — целевая частота [Гц]
        freq_min : float — нижняя граница частотной сетки [Гц]
        freq_max : float — верхняя граница частотной сетки [Гц]
        n_freqs  : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная амплитуда, ≥0.
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T = self.shape
        fi      = int(np.argmin(np.abs(self._cwt_freqs - freq_hz)))
        return self._cwt_mag[:, fi, :].reshape(X, Y, T).astype(np.float32)

    def q_factor_proxy(self, dt_ref_ms: float = 100.0,
                    freq_min: float = 5.0, freq_max: float = 120.0,
                    n_freqs: int = 64) -> np.ndarray:
        """
        Проксирует затухание (обратный Q) методом spectral ratio: сравнивает
        CWT-спектр на отсчёте t с спектром на t − dt_ref (более раннее/мелкое
        время = опорное, менее затухшее).

        dt_ref_ms : float — временной сдвиг до опорного окна [мс]
        freq_min  : float — нижняя граница частотной сетки [Гц]
        freq_max  : float — верхняя граница частотной сетки [Гц]
        n_freqs   : int — число точек частотной сетки

        Возвращает np.ndarray, shape (X, Y, T), float32 — наклон ln(A(f,t)/A(f,t_ref))
        по частоте, нормированный на dt (~ -π/Q), [1/с]. Более отрицательные значения =
        сильнее затухание между опорным и текущим отсчётом.
        """
        self._ensure_cwt(freq_min, freq_max, n_freqs)
        X, Y, T = self.shape
        shift   = int(round(dt_ref_ms / self.dt_ms))   # сдвиг в отсчётах

        mag = self._cwt_mag                             # (X*Y, n_freqs, T)
        log_mag = np.log(np.maximum(mag, 1e-8))

        # опорное окно = log_mag сдвинутый на `shift` отсчётов назад по T
        ref = np.concatenate(
            [np.repeat(log_mag[:, :, :1], shift, axis=2),  # паддинг началом
            log_mag[:, :, :-shift]], axis=2
        )
        ratio = log_mag - ref                           # ln(A(f,t)/A(f,t_ref))

        f          = self._cwt_freqs
        f_centered = (f - f.mean())[np.newaxis, :, np.newaxis]
        denom      = np.sum((f - f.mean())**2)

        slope = (f_centered * ratio).sum(axis=1) / denom / (dt_ref_ms / 1000.0)  # (X*Y, T)
        return slope.reshape(X, Y, T).astype(np.float32)

    # ── Амплитудные ───────────────────────────────────────────────────────────

    def rms_amplitude(self, window_t: int = 5) -> np.ndarray:
        """
        RMS-амплитуда в скользящем окне по времени.

        window_t : int — полуразмер окна по времени [сэмплы], итоговое окно (2·window_t+1)

        Возвращает np.ndarray, shape (X, Y, T), float32, в единицах амплитуды куба, ≥0.
        """
        size = [1, 1, 2*window_t+1]
        return np.sqrt(uniform_filter(self.cube**2, size=size)).astype(np.float32)

    def total_energy(self, window_t: int = 5) -> np.ndarray:
        """
        Суммарная энергия сигнала в скользящем окне по времени.

        window_t : int — полуразмер окна по времени [сэмплы], итоговое окно (2·window_t+1)

        Возвращает np.ndarray, shape (X, Y, T), float32, в единицах (амплитуда куба)², ≥0.
        """
        size = [1, 1, 2*window_t+1]
        N    = 2*window_t + 1
        return (uniform_filter(self.cube**2, size=size) * N).astype(np.float32)

    def rai(self) -> np.ndarray:
        """
        Relative Acoustic Impedance — интеграл (кумулятивная сумма) центрированной
        трассы по времени.

        Входных параметров нет — использует self.cube.

        Возвращает np.ndarray, shape (X, Y, T), float32, в единицах (амплитуда куба)·сэмпл.
        """
        cube_centered = self.cube - self.cube.mean(axis=2, keepdims=True)
        return np.cumsum(cube_centered, axis=2).astype(np.float32)

    # ── GLCM-атрибуты (текстура) ────────────────────────────────────────────────

    def glcm_contrast(self, axes=(0,2), window=5, offset=1, n_levels=8):
        """
        Haralick contrast — Σ p(i,j)·(i-j)².

        axes     : (0,2)=X-T (вертикальный разрез), (1,2)=Y-T, (0,1)=X-Y (карта)
        window   : int — размер окна вдоль каждой из двух осей плоскости [сэмплы]
        offset   : int — смещение для пары co-occurrence, вдоль axes[1] [сэмплы]
        n_levels : int — число уровней квантования амплитуды (см. _glcm_stats())

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, ≥0: высокие
        значения → резкие локальные контрасты уровней (разрывы рефлекторов,
        хаотичная текстура).
        """
        self._ensure_glcm(axes, window, offset, n_levels)
        return self._glcm_cache['glcm_contrast']

    def glcm_homogeneity(self, axes=(0,2), window=5, offset=1, n_levels=8):
        """
        Haralick homogeneity — Σ p(i,j)/(1+(i-j)²).

        axes     : (0,2)=X-T (вертикальный разрез), (1,2)=Y-T, (0,1)=X-Y (карта)
        window   : int — размер окна вдоль каждой из двух осей плоскости [сэмплы]
        offset   : int — смещение для пары co-occurrence, вдоль axes[1] [сэмплы]
        n_levels : int — число уровней квантования амплитуды (см. _glcm_stats())

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, диапазон (0, 1]:
        высокие значения → гладкая однородная текстура (плотный слоистый матрикс).
        """
        self._ensure_glcm(axes, window, offset, n_levels)
        return self._glcm_cache['glcm_homogeneity']

    def glcm_energy(self, axes=(0,2), window=5, offset=1, n_levels=8):
        """
        Haralick energy (angular second moment) — Σ p(i,j)².

        axes     : (0,2)=X-T (вертикальный разрез), (1,2)=Y-T, (0,1)=X-Y (карта)
        window   : int — размер окна вдоль каждой из двух осей плоскости [сэмплы]
        offset   : int — смещение для пары co-occurrence, вдоль axes[1] [сэмплы]
        n_levels : int — число уровней квантования амплитуды (см. _glcm_stats())

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, диапазон (0, 1]:
        высокие значения → мало доминирующих пар уровней, регулярная повторяющаяся текстура.
        """
        self._ensure_glcm(axes, window, offset, n_levels)
        return self._glcm_cache['glcm_energy']

    def glcm_entropy(self, axes=(0,2), window=5, offset=1, n_levels=8):
        """
        Haralick entropy — -Σ p(i,j)·ln(p(i,j)).

        axes     : (0,2)=X-T (вертикальный разрез), (1,2)=Y-T, (0,1)=X-Y (карта)
        window   : int — размер окна вдоль каждой из двух осей плоскости [сэмплы]
        offset   : int — смещение для пары co-occurrence, вдоль axes[1] [сэмплы]
        n_levels : int — число уровней квантования амплитуды (см. _glcm_stats())

        Возвращает np.ndarray, shape (X, Y, T), float32, безразмерная, ≥0: высокие
        значения → хаотичная/неупорядоченная текстура (потенциальные каверны/трещины).
        """
        self._ensure_glcm(axes, window, offset, n_levels)
        return self._glcm_cache['glcm_entropy']

    # =========================================================================
    # compute() — единая точка входа
    # =========================================================================

    def compute(self, attrs) -> dict:
        """
        Вычисляет только запрошенные атрибуты.

        attrs : list[str] | 'all' — список имён атрибутов или строка 'all'.
                Параметрические имена: 'magnitude_at_{f}hz' (например 'magnitude_at_30hz'),
                'q_factor_{dt}ms' (например 'q_factor_50ms').

        Возвращает dict[str, np.ndarray], каждый массив shape (X, Y, T), float32 —
        см. докстринг соответствующего метода-атрибута.

        Примеры
        -------
        sa.compute('all')
        sa.compute(['coherence', 'magnitude_at_30hz', 'q_factor_50ms'])
        """
        if attrs == 'all':
            attrs = list(AVAILABLE_ATTRS)

        result = {}
        for name in attrs:
            if name in AVAILABLE_ATTRS:
                result[name] = getattr(self, name)()

            elif name.startswith('magnitude_at_') and name.endswith('hz'):
                # magnitude_at_30hz → freq=30.0
                freq_hz = float(name[len('magnitude_at_'):-len('hz')])
                result[name] = self.magnitude_at(freq_hz)

            elif name.startswith('q_factor_') and name.endswith('ms'):
                # q_factor_30ms → dt_ref_ms=30.0
                dt_ref = float(name[len('q_factor_'):-len('ms')])
                result[name] = self.q_factor_proxy(dt_ref_ms=dt_ref)

            else:
                raise ValueError(
                    f"Неизвестный атрибут: '{name}'.\n"
                    f"Доступные: {AVAILABLE_ATTRS}\n"
                    f"Параметрические: magnitude_at_{{f}}hz, q_factor_{{dt}}ms, "
                    f"например 'magnitude_at_30hz', 'q_factor_50ms'"
                )
        return result

    # =========================================================================
    # Обратная совместимость с dataset.ipynb
    # =========================================================================

    def compute_all(self, dt_ms: float = None) -> dict:
        """
        Вычисляет все непараметрические атрибуты плюс magnitude_at на фиксированной
        гребёнке частот. Обратная совместимость с v1.

        dt_ms : float | None — если передан, обновляет self.dt_ms перед расчётом

        Возвращает dict[str, np.ndarray], каждый массив shape (X, Y, T), float32 —
        все атрибуты из AVAILABLE_ATTRS плюс 'magnitude_at_10hz'...'magnitude_at_60hz'
        (шаг 10 Гц).
        """
        if dt_ms is not None:
            self.dt_ms = dt_ms
        fixed_freqs = [10, 20, 30, 40, 50, 60]
        fixed_q = [10, 30, 60]

        result = self.compute('all')
        for f in fixed_freqs:
            result[f'magnitude_at_{f}hz'] = self.magnitude_at(float(f))
        for f in fixed_q:
            result[f'q_factor_{f}ms'] = self.q_factor_proxy(dt_ref_ms=float(f))

        return result
