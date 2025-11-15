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
            int topMargin = 40;    // теперь он точно будет заметен
            if (top + topMargin < bottom)
                top += topMargin;

            int bottomMargin = 40;
            if (bottom - bottomMargin > top)
                bottom -= bottomMargin;

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

    }
}
