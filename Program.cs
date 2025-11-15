using System;
using System.IO;
using System.Linq;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace PhotoSheetProcessor
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string inputPath = @"D:\Temp\input.jpg";
            string outputPath = @"D:\Temp\output_sheet.jpg";

            if (!File.Exists(inputPath))
            {
                Console.WriteLine("Файл не найден: " + inputPath);
                Console.ReadKey();
                return;
            }

            using var original = CvInvoke.Imread(inputPath);

            if (original.IsEmpty)
            {
                Console.WriteLine("Не удалось прочитать изображение.");
                Console.ReadKey();
                return;
            }

            // 1) Находим границы листа по яркости
            Rectangle pageRect = FindPageByBrightness(original);

            // Если что-то пошло не так – просто работаем с исходником
            if (pageRect.Width <= 0 || pageRect.Height <= 0)
                pageRect = new Rectangle(0, 0, original.Width, original.Height);

            using var page = new Mat(original, pageRect);

            // 2) Поворачиваем в портрет, если нужно
            Mat sheet = page.Clone();
            if (sheet.Width > sheet.Height)
            {
                var rotated = new Mat();
                CvInvoke.Rotate(sheet, rotated, RotateFlags.Rotate90CounterClockwise);
                sheet.Dispose();
                sheet = rotated;
            }

            // 3) Разворачиваем так, чтобы «верх» был наверху
            sheet = EnsureUpright(sheet);
            sheet = ApplyFinalMargins(sheet);

            CvInvoke.Imwrite(outputPath, sheet);
            sheet.Dispose();

            Console.WriteLine("Готово. Сохранено как " + outputPath);
            Console.ReadKey();
        }

        /// <summary>
        /// Находит примерные границы листа по яркости (белая бумага),
        /// обрезает снизу вторую таблицу.
        /// </summary>
        private static Rectangle FindPageByBrightness(Mat src)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(src, gray, ColorConversion.Bgr2Gray);

            using var binary = new Mat();
            // Бумага -> белый (255), фон -> чёрный
            CvInvoke.Threshold(gray, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            int h = binary.Rows;
            int w = binary.Cols;

            var data = (byte[,])binary.GetData();
            int[] rowSums = new int[h];
            int[] colSums = new int[w];

            // 1) Считаем кол-во белых пикселей по столбцам (для left/right)
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    if (data[y, x] == 255)
                    {
                        colSums[x]++;
                    }
                }
            }

            // --- сначала находим left/right как раньше ---

            int maxCol = colSums.Max();
            double colFrac = 0.80;

            int left = 0, right = w - 1;

            for (int x = 0; x < w; x++)
            {
                if (colSums[x] >= colFrac * maxCol)
                {
                    left = x;
                    break;
                }
            }

            for (int x = w - 1; x >= 0; x--)
            {
                if (colSums[x] >= colFrac * maxCol)
                {
                    right = x;
                    break;
                }
            }

            // 2) А теперь считаем rowSums ТОЛЬКО в диапазоне [left; right]
            //    – игнорируем левую/правую "лишнюю" бумагу
            for (int y = 0; y < h; y++)
            {
                int sum = 0;
                for (int x = left; x <= right; x++)
                {
                    if (data[y, x] == 255)
                        sum++;
                }
                rowSums[y] = sum;
            }

            int maxRow = rowSums.Max();
            double rowFrac = 0.90;

            int top = 0, bottom = h - 1;

            // верх
            for (int y = 0; y < h; y++)
            {
                if (rowSums[y] >= rowFrac * maxRow)
                {
                    top = y;
                    break;
                }
            }

            // низ
            for (int y = h - 1; y >= 0; y--)
            {
                if (rowSums[y] >= rowFrac * maxRow)
                {
                    bottom = y;
                    break;
                }
            }

            // Дополнительный отступ сверху/снизу:
            int topMargin = 40;    // базовая подрезка верха, остальное делаем после выравнивания
            if (top + topMargin < bottom)
                top += topMargin;

            int bottomMargin = 40;
            if (bottom - bottomMargin > top)
                bottom -= bottomMargin;

            int rightMargin = 20;  // лёгкая подрезка справа
            if (right - rightMargin > left)
                right -= rightMargin;

            // Финальный прямоугольник
            int width = Math.Max(1, right - left + 1);
            int height = Math.Max(1, bottom - top + 1);

            return new Rectangle(left, top, width, height);
        }

        private static Mat EnsureUpright(Mat sheet)
        {
            // 1) Лист уже портретный, но на всякий случай можно ещё раз проверить
            if (sheet.Width > sheet.Height)
            {
                var rot = new Mat();
                CvInvoke.Rotate(sheet, rot, RotateFlags.Rotate90CounterClockwise);
                sheet.Dispose();
                sheet = rot;
            }

            var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);

            var binary = new Mat();
            // Инверсия: «чернила» -> белый, фон -> чёрный
            CvInvoke.Threshold(gray, binary, 0, 255,
                ThresholdType.BinaryInv | ThresholdType.Otsu);

            int h = binary.Rows;
            int w = binary.Cols;
            int bandH = Math.Max(1, h / 5); // 20% сверху/снизу

            var topRect = new Rectangle(0, 0, w, bandH);
            var bottomRect = new Rectangle(0, h - bandH, w, bandH);

            using var top = new Mat(binary, topRect);
            using var bottom = new Mat(binary, bottomRect);

            double topInk = CvInvoke.CountNonZero(top);
            double bottomInk = CvInvoke.CountNonZero(bottom);

            gray.Dispose();
            binary.Dispose();

            // В правильной ориентации вверху больше текста/цифр
            if (bottomInk > topInk)
            {
                var rot180 = new Mat();
                CvInvoke.Rotate(sheet, rot180, RotateFlags.Rotate180);
                sheet.Dispose();
                sheet = rot180;
            }

            return sheet;
        }

        private static Mat ApplyFinalMargins(Mat sheet)
        {
            int extraTop = DetectContentTop(sheet);
            int extraRight = 30;   // лёгкая подрезка справа

            int left = 0;
            int top = Math.Min(extraTop, Math.Max(0, sheet.Height - 1));

            int width = Math.Max(1, sheet.Width - extraRight - left);
            int height = Math.Max(1, sheet.Height - top);

            var rect = new Rectangle(left, top, width, height);
            var cropped = new Mat(sheet, rect).Clone();

            sheet.Dispose();
            return cropped;
        }

        private static int DetectContentTop(Mat sheet)
        {
            int inkTop = DetectTopByInk(sheet);
            if (inkTop > 0)
                return inkTop;

            int edgeTop = DetectTopByEdge(sheet);
            if (edgeTop > 0)
            {
                int marginBelowEdge = 18;
                return Math.Max(0, edgeTop + marginBelowEdge);
            }

            return 0;
        }

        private static int DetectTopByInk(Mat sheet)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);

            using var blurred = new Mat();
            CvInvoke.GaussianBlur(gray, blurred, new Size(3, 3), 0);

            using var binary = new Mat();
            CvInvoke.Threshold(blurred, binary, 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu);

            using var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            CvInvoke.MorphologyEx(binary, binary, MorphOp.Dilate, kernel, new Point(-1, -1), 1, BorderType.Default, default(MCvScalar));

            int h = binary.Rows;
            int w = binary.Cols;

            int marginX = w / 8;
            int roiLeft = Math.Max(0, marginX);
            int roiRight = Math.Min(w, w - marginX);
            if (roiRight <= roiLeft)
            {
                roiLeft = 0;
                roiRight = w;
            }

            int roiWidth = roiRight - roiLeft;
            var roiRect = new Rectangle(roiLeft, 0, roiWidth, h);

            using var roi = new Mat(binary, roiRect);
            var data = (byte[,])roi.GetData();

            int[] rowInk = new int[roi.Rows];
            for (int y = 0; y < roi.Rows; y++)
            {
                int inkCount = 0;
                for (int x = 0; x < roiWidth; x++)
                {
                    if (data[y, x] > 0)
                        inkCount++;
                }
                rowInk[y] = inkCount;
            }

            int minInkPerRow = Math.Max(2, (int)(roiWidth * 0.002));
            int lookAhead = 4;
            int safetyMargin = 6;

            for (int y = 0; y < rowInk.Length; y++)
            {
                if (rowInk[y] < minInkPerRow)
                    continue;

                bool stable = true;
                for (int k = 1; k <= lookAhead && y + k < rowInk.Length; k++)
                {
                    if (rowInk[y + k] < minInkPerRow / 2)
                    {
                        stable = false;
                        break;
                    }
                }

                if (stable)
                    return Math.Max(0, y - safetyMargin);
            }

            return 0;
        }

        private static int DetectTopByEdge(Mat sheet)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);

            using var blurred = new Mat();
            CvInvoke.GaussianBlur(gray, blurred, new Size(5, 5), 0);

            using var edges = new Mat();
            CvInvoke.Canny(blurred, edges, 40, 120);

            int h = edges.Rows;
            int w = edges.Cols;

            int marginX = w / 6;
            int roiLeft = Math.Max(0, marginX);
            int roiRight = Math.Min(w, w - marginX);
            if (roiRight <= roiLeft)
            {
                roiLeft = 0;
                roiRight = w;
            }

            int roiWidth = roiRight - roiLeft;
            var roiRect = new Rectangle(roiLeft, 0, roiWidth, h);

            using var roi = new Mat(edges, roiRect);
            var data = (byte[,])roi.GetData();

            int[] rowEdges = new int[roi.Rows];
            for (int y = 0; y < roi.Rows; y++)
            {
                int edgeCount = 0;
                for (int x = 0; x < roiWidth; x++)
                {
                    if (data[y, x] > 0)
                        edgeCount++;
                }
                rowEdges[y] = edgeCount;
            }

            int minEdgesPerRow = Math.Max(3, (int)(roiWidth * 0.003));
            int runLength = 3;

            for (int y = 0; y < rowEdges.Length; y++)
            {
                if (rowEdges[y] < minEdgesPerRow)
                    continue;

                bool run = true;
                for (int k = 1; k < runLength && y + k < rowEdges.Length; k++)
                {
                    if (rowEdges[y + k] < minEdgesPerRow / 2)
                    {
                        run = false;
                        break;
                    }
                }

                if (run)
                    return y;
            }

            return 0;
        }

    }
}
