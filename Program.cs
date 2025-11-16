using System;
using System.IO;
using System.Linq;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace PhotoSheetProcessor
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string dataDirectory = @"D:\Data";

            if (!Directory.Exists(dataDirectory))
            {
                Console.WriteLine("Папка не найдена: " + dataDirectory);
                Console.ReadKey();
                return;
            }

            var jpgFiles = Directory
                .EnumerateFiles(dataDirectory, "*.jpg", SearchOption.TopDirectoryOnly)
                .ToList();

            if (jpgFiles.Count == 0)
            {
                Console.WriteLine("В папке " + dataDirectory + " нет файлов .jpg.");
                Console.ReadKey();
                return;
            }

            int processed = 0;
            foreach (var inputPath in jpgFiles)
            {
                string fileName = Path.GetFileNameWithoutExtension(inputPath) ?? "output";
                string outputPath = Path.Combine(dataDirectory, fileName + "_new.jpg");

                try
                {
                    if (ProcessFile(inputPath, outputPath))
                    {
                        processed++;
                        Console.WriteLine("Готово. Сохранено как " + outputPath);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Ошибка при обработке файла " + inputPath + ": " + ex.Message);
                }
            }

            Console.WriteLine("Всего обработано файлов: " + processed);
            Console.ReadKey();
        }

        private static bool ProcessFile(string inputPath, string outputPath)
        {
            if (!File.Exists(inputPath))
            {
                Console.WriteLine("Файл не найден: " + inputPath);
                return false;
            }

            using var original = CvInvoke.Imread(inputPath);

            if (original.IsEmpty)
            {
                Console.WriteLine("Не удалось прочитать изображение: " + inputPath);
                return false;
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
            return true;
        }

        /// <summary>
        /// Находит примерные границы листа по яркости (белая бумага),
        /// обрезает снизу вторую таблицу.
        /// </summary>
        private static int _fallbackTopReserve = 0;

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

            if (bottom <= top)
            {
                top = 0;
                bottom = h - 1;
            }

            int bottomMargin = 40;
            if (bottom - bottomMargin > top)
                bottom -= bottomMargin;

            int rightMargin = 20;  // лёгкая подрезка справа
            if (right - rightMargin > left)
                right -= rightMargin;

            // Финальный прямоугольник
            int width = Math.Max(1, right - left + 1);
            int height = Math.Max(1, bottom - top + 1);

            if (bottom > top && top > 0)
            {
                int reserve = (bottom - top) / 200;
                _fallbackTopReserve = Math.Min(3, Math.Max(2, reserve));
            }
            else
            {
                _fallbackTopReserve = 0;
            }

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
            int detectedTop = DetectContentTop(sheet);
            int extraRight = 30;   // лёгкая подрезка справа

            int left = 0;
            int top;
            if (detectedTop >= 0)
            {
                top = Math.Min(detectedTop, Math.Max(0, sheet.Height - 1));
            }
            else
            {
                top = Math.Min(_fallbackTopReserve, Math.Max(0, sheet.Height - 1));
            }

            int width = Math.Max(1, sheet.Width - extraRight - left);
            int height = Math.Max(1, sheet.Height - top);

            var rect = new Rectangle(left, top, width, height);
            var cropped = new Mat(sheet, rect).Clone();

            sheet.Dispose();
            return cropped;
        }

        private static int DetectContentTop(Mat sheet)
        {
            int lineTop = DetectTopByHorizontalLine(sheet);
            if (lineTop >= 0)
                return lineTop;

            int inkTop = DetectTopByInk(sheet);
            if (inkTop > 0)
                return inkTop;

            int edgeTop = DetectTopByEdge(sheet);
            if (edgeTop > 0)
            {
                int marginBelowEdge = 18;
                return Math.Max(0, edgeTop + marginBelowEdge);
            }

            return -1;
        }

        private static int DetectTopByHorizontalLine(Mat sheet)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);

            using var inkMask = new Mat();
            CvInvoke.Threshold(gray, inkMask, 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu);

            using var lineMask = inkMask.Clone();

            int kernelWidth = Math.Max(10, sheet.Width / 4);
            int kernelHeight = Math.Min(5, Math.Max(3, sheet.Height / 200));
            using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Rectangle, new Size(kernelWidth, kernelHeight), new Point(-1, -1));
            CvInvoke.MorphologyEx(lineMask, lineMask, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, default(MCvScalar));

            int roiHeight = Math.Max(1, sheet.Height / 3);
            var roiRect = new Rectangle(0, 0, lineMask.Cols, roiHeight);
            using var roi = new Mat(lineMask, roiRect);

            LineSegment2D[] lineSegments = CvInvoke.HoughLinesP(
                roi,
                1,
                Math.PI / 180,
                80,
                Math.Max(1, (int)(roiRect.Width * 0.7)),
                10);

            int bestTop = int.MaxValue;
            foreach (var segment in lineSegments)
            {
                int x1 = segment.P1.X;
                int y1 = segment.P1.Y;
                int x2 = segment.P2.X;
                int y2 = segment.P2.Y;

                double angle = Math.Abs(Math.Atan2(y2 - y1, x2 - x1) * 180.0 / Math.PI);
                if (angle > 5)
                    continue;

                double length = segment.Length;
                if (length < roiRect.Width * 0.7)
                    continue;

                int segmentTop = Math.Min(y1, y2);
                if (segmentTop < bestTop)
                    bestTop = segmentTop;
            }

            if (bestTop == int.MaxValue)
                return -1;

            int absoluteLineTop = bestTop + roiRect.Top;

            int searchTop = Math.Max(0, absoluteLineTop - sheet.Height / 4);
            int searchHeight = Math.Max(1, absoluteLineTop - searchTop);
            int marginX = sheet.Width / 10;
            int textLeft = Math.Max(0, marginX);
            int textWidth = Math.Max(1, sheet.Width - textLeft * 2);
            if (textLeft + textWidth > sheet.Width)
                textWidth = sheet.Width - textLeft;

            var textRoiRect = new Rectangle(textLeft, searchTop, textWidth, searchHeight);
            if (textRoiRect.Bottom > inkMask.Rows)
                textRoiRect.Height = Math.Max(1, inkMask.Rows - textRoiRect.Top);

            int firstInkRow = -1;
            using (var textRoi = new Mat(inkMask, textRoiRect))
            {
                var data = (byte[,])textRoi.GetData();
                int minInkPerRow = Math.Max(3, textRoi.Cols / 60);
                for (int y = 0; y < textRoi.Rows; y++)
                {
                    int inkCount = 0;
                    for (int x = 0; x < textRoi.Cols; x++)
                    {
                        if (data[y, x] > 0)
                            inkCount++;
                    }

                    if (inkCount >= minInkPerRow)
                    {
                        firstInkRow = y;
                        break;
                    }
                }
            }

            int topCandidate = firstInkRow >= 0 ? textRoiRect.Top + firstInkRow : absoluteLineTop;
            int reserve = 3;
            int absoluteTop = Math.Max(0, topCandidate - reserve);
            return absoluteTop;
        }

        private static int DetectTopByInk(Mat sheet)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);

            using var blurred = new Mat();
            CvInvoke.GaussianBlur(gray, blurred, new Size(3, 3), 0);

            using var binary = new Mat();
            CvInvoke.Threshold(blurred, binary, 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu);

            using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Rectangle, new Size(3, 3), new Point(-1, -1));
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
