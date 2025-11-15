using System;
using System.Collections.Generic;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using System.Drawing;

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

            var sheet = ExtractSheet(original);

            if (sheet == null || sheet.IsEmpty)
            {
                Console.WriteLine("Не удалось найти лист на изображении.");
                Console.ReadKey();
                return;
            }

            sheet = EnsureUpright(sheet);           // 2) привести к правильной ориентации
            sheet = TightCropByInk(sheet);          // 3) обрезать по содержимому

            CvInvoke.Imwrite(outputPath, sheet);
            sheet.Dispose();

            Console.WriteLine("Готово. Сохранено как " + outputPath);
            Console.ReadKey();
        }

        /// <summary>
        /// Ищет самый крупный четырёхугольный контур (лист) и возвращает выпрямленное изображение.
        /// </summary>
        private static Mat? ExtractSheet(Mat input)
        {
            // В серый и размытие
            var gray = new Mat();
            CvInvoke.CvtColor(input, gray, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(gray, gray, new Size(5, 5), 0);

            // Границы
            var edges = new Mat();
            CvInvoke.Canny(gray, edges, 50, 150);

            // Немного расширим границы, чтобы контур листа не прерывался
            var kernel = CvInvoke.GetStructuringElement(
                MorphShapes.Rectangle,
                new Size(7, 7),
                new Point(-1, -1));
            CvInvoke.Dilate(edges, edges, kernel, new Point(-1, -1), 2, BorderType.Reflect, default);

            // --- Для отладки можно сохранить промежуточное изображение ---
            // CvInvoke.Imwrite(@"D:\Temp\edges.jpg", edges);

            // Поиск контуров
            using var contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(edges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            double maxArea = 0;
            RotatedRect bestRect = new RotatedRect();
            bool found = false;

            for (int i = 0; i < contours.Size; i++)
            {
                using var contour = contours[i];
                double area = CvInvoke.ContourArea(contour);
                if (area < 10000) // отсечь совсем мелкий мусор
                    continue;

                // Минимальный повёрнутый прямоугольник вокруг контура
                var rect = CvInvoke.MinAreaRect(contour);
                double rectArea = rect.Size.Width * rect.Size.Height;

                if (rectArea > maxArea)
                {
                    maxArea = rectArea;
                    bestRect = rect;
                    found = true;
                }
            }

            if (!found)
            {
                gray.Dispose();
                edges.Dispose();
                kernel.Dispose();
                return null;
            }

            // 4 вершины прямоугольника
            PointF[] pts = bestRect.GetVertices(); // уже 4 точки

            // Упорядочим точки (лево-верх, право-верх, право-низ, лево-низ)
            PointF[] ordered = OrderPoints(pts);

            // Размер целевого прямоугольника
            float widthA = Distance(ordered[2], ordered[3]);
            float widthB = Distance(ordered[1], ordered[0]);
            float maxWidth = Math.Max(widthA, widthB);

            float heightA = Distance(ordered[1], ordered[2]);
            float heightB = Distance(ordered[0], ordered[3]);
            float maxHeight = Math.Max(heightA, heightB);

            var dst = new[]
            {
        new PointF(0, 0),
        new PointF(maxWidth - 1, 0),
        new PointF(maxWidth - 1, maxHeight - 1),
        new PointF(0, maxHeight - 1)
    };

            using var srcMat = new Matrix<float>(new[,] {
        { ordered[0].X, ordered[0].Y },
        { ordered[1].X, ordered[1].Y },
        { ordered[2].X, ordered[2].Y },
        { ordered[3].X, ordered[3].Y }
    });

            using var dstMat = new Matrix<float>(new[,] {
        { dst[0].X, dst[0].Y },
        { dst[1].X, dst[1].Y },
        { dst[2].X, dst[2].Y },
        { dst[3].X, dst[3].Y }
    });

            var M = CvInvoke.GetPerspectiveTransform(srcMat, dstMat);
            var warped = new Mat();
            CvInvoke.WarpPerspective(input, warped, M, new Size((int)maxWidth, (int)maxHeight));

            // Очистка временных матриц
            gray.Dispose();
            edges.Dispose();
            kernel.Dispose();

            return warped;
        }


        private static PointF[] OrderPoints(PointF[] pts)
        {
            // 4 точки
            var result = new PointF[4];

            // Суммы и разности координат
            float minSum = float.MaxValue, maxSum = float.MinValue;
            float minDiff = float.MaxValue, maxDiff = float.MinValue;

            foreach (var p in pts)
            {
                float sum = p.X + p.Y;
                float diff = p.X - p.Y;

                if (sum < minSum) { minSum = sum; result[0] = p; } // top-left
                if (sum > maxSum) { maxSum = sum; result[2] = p; } // bottom-right
                if (diff < minDiff) { minDiff = diff; result[3] = p; } // bottom-left
                if (diff > maxDiff) { maxDiff = diff; result[1] = p; } // top-right
            }

            return result;
        }

        private static float Distance(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        private static List<Mat> MakePortrait(Mat sheet)
        {
            var candidates = new List<Mat>();
            if (sheet.Width > sheet.Height)
            {
                var clockwise = new Mat();
                CvInvoke.Rotate(sheet, clockwise, RotateFlags.Rotate90Clockwise);
                candidates.Add(clockwise);

                var counterClockwise = new Mat();
                CvInvoke.Rotate(sheet, counterClockwise, RotateFlags.Rotate90CounterClockwise);
                candidates.Add(counterClockwise);

                sheet.Dispose();
            }
            else
            {
                candidates.Add(sheet);
            }
            return candidates;
        }

        private static Mat EnsureUpright(Mat sheet)
        {
            // 1. Убедимся, что портретная ориентация
            var candidates = MakePortrait(sheet);

            (double topInk, double bottomInk) MeasureInk(Mat candidate)
            {
                using var gray = new Mat();
                CvInvoke.CvtColor(candidate, gray, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(gray, gray, 0, 255, ThresholdType.BinaryInv | ThresholdType.Otsu);

                int h = gray.Rows;
                int w = gray.Cols;
                int bandH = Math.Max(1, h / 5);

                var topRect = new Rectangle(0, 0, w, bandH);
                var bottomRect = new Rectangle(0, h - bandH, w, bandH);

                using var top = new Mat(gray, topRect);
                using var bottom = new Mat(gray, bottomRect);

                double topInk = CvInvoke.CountNonZero(top);
                double bottomInk = CvInvoke.CountNonZero(bottom);

                return (topInk, bottomInk);
            }

            Mat? selected = null;
            double selectedTopInk = double.MinValue;

            foreach (var candidate in candidates)
            {
                var (topInk, _) = MeasureInk(candidate);
                if (selected == null || topInk > selectedTopInk)
                {
                    selected?.Dispose();
                    selected = candidate;
                    selectedTopInk = topInk;
                }
                else
                {
                    candidate.Dispose();
                }
            }

            if (selected == null)
            {
                throw new InvalidOperationException("Не удалось получить портретную ориентацию.");
            }

            sheet = selected;

            // 2. Измерим распределение «чернил» для выбранной ориентации
            var (finalTopInk, finalBottomInk) = MeasureInk(sheet);

            // Хотим, чтобы наверху было больше чернил, чем внизу.
            if (finalBottomInk > finalTopInk)
            {
                var rot180 = new Mat();
                CvInvoke.Rotate(sheet, rot180, RotateFlags.Rotate180);
                sheet.Dispose();
                sheet = rot180;
            }
            return sheet;
        }

        private static Mat TightCropByInk(Mat sheet)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(sheet, gray, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(gray, gray, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            // Инвертируем, чтобы «чернила» были белыми (1), фон — чёрным
            CvInvoke.BitwiseNot(gray, gray);

            using var points = new VectorOfPoint();
            CvInvoke.FindNonZero(gray, points);

            if (points.Size == 0)
            {
                // На всякий случай — если вдруг ничего не найдено
                gray.Dispose();
                return sheet;
            }

            Rectangle bbox = CvInvoke.BoundingRectangle(points);

            // Добавим небольшой отступ (например, 10 пикселей)
            int margin = 10;
            bbox.X = Math.Max(0, bbox.X - margin);
            bbox.Y = Math.Max(0, bbox.Y - margin);
            bbox.Width = Math.Min(sheet.Width - bbox.X, bbox.Width + 2 * margin);
            bbox.Height = Math.Min(sheet.Height - bbox.Y, bbox.Height + 2 * margin);

            var cropped = new Mat(sheet, bbox);

            gray.Dispose();
            return cropped;
        }



    }


}
