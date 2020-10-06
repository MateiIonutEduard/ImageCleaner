using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using OpenCL.Net;
using System.Diagnostics;

#pragma warning disable

namespace ImageCleaner
{
    public partial class Form1 : Form
    {
        CommandQueue queue;
        private Context context;
        private Device device;

        string path = string.Empty;
        string script = string.Empty;

        PointF O = PointF.Empty;
        float radius = 0.0f;
        int count = 0;

        List<PointF> map = new List<PointF>();
        Random rand = new Random(System.Environment.TickCount);

        string newPath = string.Empty;
        List<PointF> points = new List<PointF>();
        Image hdc = null, vdc = null;

        public Form1()
        {
            InitializeComponent();
            hdc = pictureBox1.Image;
            vdc = pictureBox1.Image;

            script = File.ReadAllText("shaders/fix.cl");
            path = @"data/init.bmp";
            SetupDevice();
        }

        private void SetupDevice()
        {
            Event e;
            ErrorCode error;
            Platform[] platforms = Cl.GetPlatformIDs(out error);
            Device[] devices = Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out error);
            device = devices[0];
            context = Cl.CreateContext(null, 1, devices, null, IntPtr.Zero, out error);
            queue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out error);
        }

        private void BuildPoints()
        {
            int len = 10240 * count;
            int maxr = (int)Math.Floor(2 * radius);

            for (int i = 1; i <= len; i++)
            {
                PointF point = new PointF
                {
                    X = ((rand.Next() % (2 * maxr)) - maxr) + O.X,
                    Y = ((rand.Next() % (2 * maxr)) - maxr) + O.Y
                };

                map.Add(point);
            }
        }

        private void FixImage()
        {
            Event e;
            ErrorCode error;
            OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new[] { script }, null, out error);
            error = Cl.BuildProgram(program, 0, null, string.Empty, null, IntPtr.Zero);
            //MessageBox.Show(error.ToString());
            Kernel kernel = Cl.CreateKernel(program, "fixImage", out error);
            int intPtrSize = Marshal.SizeOf(typeof(IntPtr));

            Mem dest;
            OpenCL.Net.ImageFormat clImageFormat = new OpenCL.Net.ImageFormat(OpenCL.Net.ChannelOrder.RGBA, OpenCL.Net.ChannelType.Unsigned_Int8);
            int inputImgWidth, inputImgHeight;

            Image img = Image.FromFile(path);
            inputImgWidth = img.Width;
            inputImgHeight = img.Height;

            Bitmap bmp = new Bitmap(img);
            float[] buffer = new float[40960 * count];
            float[] array = new float[] { radius, O.X, O.Y };

            dest = (Mem)Cl.CreateBuffer(context, MemFlags.WriteOnly, new IntPtr(count * 40960 * sizeof(float)), out error);
            Mem P = (Mem)Cl.CreateBuffer(context, MemFlags.ReadWrite, Marshal.SizeOf(typeof(PointF)) * 10240 * count, out error);
            Mem data = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly, Marshal.SizeOf(typeof(PointF)) * points.Count, out error);
            Mem ptr = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly, 3 * sizeof(float), out error);

            Cl.EnqueueWriteBuffer(queue, P, Bool.True, IntPtr.Zero, new IntPtr(Marshal.SizeOf(typeof(PointF)) * 10240 * count), map.ToArray(), 0, null, out e);
            Cl.EnqueueWriteBuffer(queue, data, Bool.True, IntPtr.Zero, new IntPtr(Marshal.SizeOf(typeof(PointF)) * points.Count), points.ToArray(), 0, null, out e);
            Cl.EnqueueWriteBuffer(queue, ptr, Bool.True, IntPtr.Zero, (IntPtr)3, array, 0, null, out e);

            error = Cl.SetKernelArg(kernel, 0, (IntPtr)intPtrSize, dest);
            error |= Cl.SetKernelArg(kernel, 1, (IntPtr)intPtrSize, P);
            error |= Cl.SetKernelArg(kernel, 2, (IntPtr)intPtrSize, data);
            error |= Cl.SetKernelArg(kernel, 3, (IntPtr)intPtrSize, ptr);
            error |= Cl.SetKernelArg(kernel, 4, (IntPtr)intPtrSize, points.Count);

            IntPtr[] workGroupSizePtr = new IntPtr[] { (IntPtr)count, (IntPtr)10240 };
            Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, workGroupSizePtr, null, 0, null, out e);
            Cl.Finish(queue);


            error |= Cl.EnqueueReadBuffer(queue,
                        dest, 
                        Bool.True, 
                        IntPtr.Zero, 
                        new IntPtr(sizeof(float) * 40960 * count), 
                        buffer, 
                        0, 
                        null, 
                        out e);

            Cl.ReleaseKernel(kernel);
            Cl.ReleaseCommandQueue(queue);

            Cl.ReleaseMemObject(dest);
            Cl.ReleaseMemObject(P);
            Cl.ReleaseMemObject(data);

            for (int i = 0; i < buffer.Length; i += 4)
            {
                int dx = (int)Math.Round(buffer[i]);
                int dy = (int)Math.Round(buffer[i + 1]);

                int sx = (int)Math.Round(buffer[i + 2]);
                int sy = (int)Math.Round(buffer[i + 3]);

                bool test = (sx >= 0 && sx < img.Width) && (sy >= 0 && sy < img.Height);

                Color src = test ? bmp.GetPixel(sx, sy) : Color.White;
                if((dx >= 0 && dx < img.Width) && (dy >= 0 && dy < img.Height)) bmp.SetPixel(dx, dy, src);
            }

            int index = path.LastIndexOf('.');
            string ext = path.Substring(index + 1);
            newPath = path.Substring(0, index) + $"_final.{ext}";
            System.Drawing.Imaging.ImageFormat format = System.Drawing.Imaging.ImageFormat.Jpeg;

            if (ext == "bmp") format = System.Drawing.Imaging.ImageFormat.Bmp;
            else if (ext == "png") format = System.Drawing.Imaging.ImageFormat.Png;
            bmp.Save(newPath, format);
        }

        private PointF CenterOf()
        {
            float x = 0.0f, y = 0.0f;

            foreach(PointF point in points)
            {
                x += point.X;
                y += point.Y;
            }

            x /= points.Count;
            y /= points.Count;
            return new PointF(x, y);
        }

        private float MaxRadius()
        {
            float dmax = 0.0f;
            PointF q = CenterOf();

            foreach(PointF p in points)
            {
                float dx = p.X - q.X;
                float dy = p.Y - q.Y;

                float length = (float)Math.Sqrt(dx * dx + dy * dy);
                if (dmax < length) dmax = length;
            }

            return dmax;
        }

        private void chooseImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog root = new OpenFileDialog();
            root.Filter = "Image files(*.bmp, *.png, *.jpeg, *.jpg)|*.bmp; *.png; *.jpeg; *.jpg";
            root.DefaultExt = "*.png";

            if (root.ShowDialog() == DialogResult.OK)
            {
                hdc = Image.FromFile(root.FileName);
                vdc = Image.FromFile(root.FileName);
                pictureBox1.Image = hdc;
                path = root.FileName;

                points.Clear();
                map.Clear();
                count = 0;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            // Creates all requirements to start correction...
            
            Task.Run(() =>
            {
                O = CenterOf();
                radius = MaxRadius();
                count = points.Count >> 2;
                BuildPoints();

                Graphics g = Graphics.FromImage(hdc);
                int X = (int)Math.Floor(O.X);
                int Y = (int)Math.Floor(O.Y);

                g.FillEllipse(Brushes.Red, new Rectangle(X, Y, 10, 10));
                pictureBox1.Invalidate();
            });
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // Fix up the image and write new file to disk.
            Stopwatch sw = new Stopwatch();
            if(count == 0)
            {
                MessageBox.Show("Must to process the image first!", "App Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            Task.Run(() =>
            {
                sw.Start();
                FixImage();
                sw.Stop();
            }).ContinueWith(t =>
            {
                TimeSpan ts = sw.Elapsed;
                MessageBox.Show($"Work Complete! Total elapsed time... {ts.Minutes}:{ts.Seconds}:{ts.Milliseconds}", "Image Cleaner", MessageBoxButtons.OK, MessageBoxIcon.Information);
                Process.Start(newPath);
            });
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right) return;
            PointF dest = e.Location;

            PointF point = new PointF
            {
                X = (dest.X * hdc.Width) / pictureBox1.Width,
                Y = (dest.Y * hdc.Height) / pictureBox1.Height
            };

            points.Add(point);

            if (points.Count > 1)
            {
                Graphics g = Graphics.FromImage(hdc);
                g.DrawLine(Pens.White, points[points.Count - 1], points[points.Count - 2]);
                pictureBox1.Invalidate();
            }
        }

        private void goBackToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if(points.Count > 0) points.RemoveAt(points.Count - 1);

            Graphics g = Graphics.FromImage(hdc);
            g.DrawImage(vdc, 0, 0);

            for(int i = 0; i < points.Count - 1; i++)
                g.DrawLine(Pens.White, points[i], points[i + 1]);

            pictureBox1.Invalidate();
        }

        private void fillToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (points.Count < 2) return;
            Graphics g = Graphics.FromImage(hdc);
            points.Add(points[0]);

            for (int i = 0; i < points.Count - 1; i++)
                g.DrawLine(Pens.White, points[i], points[i + 1]);

            pictureBox1.Invalidate();
        }
    }
}
