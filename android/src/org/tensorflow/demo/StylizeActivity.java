/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.Toast;
import android.util.Log;
import android.util.DisplayMetrics;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Android app to perform inference the camera feed using a trained CycleGAN model
 * Models available at https://github.com/andrewginns/CycleGAN-TF-Android/releases
 * Based on the skeleton code from
 * https://github.com/googlecodelabs/tensorflow-style-transfer-android
 */
public class StylizeActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private TensorFlowInferenceInterface inferenceInterface;

    private static final String MODEL_FILE = "file:///android_asset/optimized-graph.pb";
    private static final String INPUT_NODE = "inputA";
    private static final String OUTPUT_NODE = "a2b_generator/output_image";

    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final float TEXT_SIZE_DIP = 12;

    private static final boolean DEBUG_MODEL = false;

    private static final int[] SIZES = {128, 192, 256, 384, 512, 720};

    private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);

    // Start at a medium size, but let the user step up through smaller sizes so they don't get
    // immediately stuck processing a large image.
    private int desiredSizeIndex = -1;
    private int desiredSize = 384;
    private int initializedSize = 0;

    private Integer sensorOrientation;

    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;

    private int[] intValues;
    private float[] floatValues;

    private int frameNum = 0;

    private Bitmap cropCopyBitmap;
    private Bitmap textureCopyBitmap;

    private boolean computing = false;

    private Matrix frameToCropTransform;

    private BorderedText borderedText;

    private long lastProcessingTimeMs;

    private final OnTouchListener gridTouchAdapter =
            new OnTouchListener() {
                ImageSlider slider = null;

                @SuppressLint("ClickableViewAccessibility")
                @Override
                public boolean onTouch(final View v, final MotionEvent event) {
                    switch (event.getActionMasked()) {
                        case MotionEvent.ACTION_DOWN:
                            Rect rect = new Rect();
                            if (rect.contains((int) event.getX(), (int) event.getY())) {
                                slider.setHighlighted(true);
                            }
                            break;

                        case MotionEvent.ACTION_MOVE:
                            if (slider != null) {
                                Rect rect2 = new Rect();
                                slider.getHitRect(rect2);
                            }
                            break;

                        case MotionEvent.ACTION_UP:
                            if (slider != null) {
                                slider.setHighlighted(false);
                                slider = null;
                            }
                            break;

                        default: // fall out

                    }
                    return true;
                }
            };

    @Override
    public void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_stylize;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private class ImageSlider extends ImageView {
        private float value;
        private boolean highlighted = false;

        private final Paint boxPaint;
        private final Paint linePaint;

        public ImageSlider(final Context context) {
            super(context);
            value = 0.0f;

            boxPaint = new Paint();
            boxPaint.setColor(Color.BLACK);
            boxPaint.setAlpha(128);

            linePaint = new Paint();
            linePaint.setColor(Color.WHITE);
            linePaint.setStrokeWidth(10.0f);
            linePaint.setStyle(Style.STROKE);
        }

        @Override
        public void onDraw(final Canvas canvas) {
            super.onDraw(canvas);
            final float y = (1.0f - value) * canvas.getHeight();

            if (value > 0.0f) {
                canvas.drawLine(0, y, canvas.getWidth(), y, linePaint);
            }

            if (highlighted) {
                canvas.drawRect(0, 0, getWidth(), getHeight(), linePaint);
            }
        }

        @Override
        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
        }

        public void setValue(final float value) {
            this.value = value;
            postInvalidate();
        }

        public void setHighlighted(final boolean highlighted) {
            this.highlighted = highlighted;
            this.postInvalidate();
        }
    }

    private class ImageGridAdapter extends BaseAdapter {
        final ArrayList<Button> buttons = new ArrayList<>();


        {
            final Button sizeButton =
                    new Button(StylizeActivity.this) {
                        @Override
                        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
                            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
                            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
                        }
                    };
            sizeButton.setText("" + desiredSize);
            sizeButton.setOnClickListener(
                    new OnClickListener() {
                        @Override
                        public void onClick(final View v) {
                            desiredSizeIndex = (desiredSizeIndex + 1) % SIZES.length;
                            desiredSize = SIZES[desiredSizeIndex];
                            sizeButton.setText("" + desiredSize);
                            sizeButton.postInvalidate();
                        }
                    });

            final Button saveButton =
                    new Button(StylizeActivity.this) {
                        @Override
                        protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
                            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
                            setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
                        }
                    };
            saveButton.setText("save");
            saveButton.setTextSize(12);

            saveButton.setOnClickListener(
                    new OnClickListener() {
                        @Override
                        public void onClick(final View v) {
                            if (textureCopyBitmap != null) {
                                ImageUtils.saveBitmap(textureCopyBitmap, "Processed" + frameNum + ".png");
                                Toast.makeText(
                                        StylizeActivity.this,
                                        "Saved image to: /sdcard/tensorflow/" + "Processed" + frameNum + ".png",
                                        Toast.LENGTH_LONG)
                                        .show();
                            }
                        }
                    });

            buttons.add(sizeButton);
            buttons.add(saveButton);
        }

        @Override
        public int getCount() {
            return buttons.size();
        }

        @Override
        public Object getItem(final int position) {
            if (position < buttons.size()) {
                return buttons.get(position);
            } else {
                return position - buttons.size();
            }
        }

        @Override
        public long getItemId(final int position) {
            return getItem(position).hashCode();
        }

        @Override
        public View getView(final int position, final View convertView, final ViewGroup parent) {
            if (convertView != null) {
                return convertView;
            }
            return (View) getItem(position);
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    //The app skeleton uses a custom camera fragment that will call this method once permissions have
    // been granted and the camera is available to use.
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        //Interface with the TFlow inference API
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();

        LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

        sensorOrientation = rotation + screenOrientation;

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderDebug(canvas);
                    }
                });

        ImageGridAdapter adapter = new ImageGridAdapter();
        GridView grid = findViewById(R.id.grid_layout);
        grid.setAdapter(adapter);
        grid.setOnTouchListener(gridTouchAdapter);

    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image image = null;

        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (computing) {
                image.close();
                return;
            }

            if (desiredSize != initializedSize) {
                LOGGER.i(
                        "Initializing at size preview size %dx%d, stylize size %d",
                        previewWidth, previewHeight, desiredSize);
                rgbBytes = new int[previewWidth * previewHeight];
                rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
                croppedBitmap = Bitmap.createBitmap(desiredSize, desiredSize, Config.ARGB_8888);

                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                desiredSize, desiredSize,
                                sensorOrientation, true);

                Matrix cropToFrameTransform = new Matrix();
                frameToCropTransform.invert(cropToFrameTransform);

                yuvBytes = new byte[3][];

                intValues = new int[desiredSize * desiredSize];
                floatValues = new float[desiredSize * desiredSize * 3];
                initializedSize = desiredSize;
            }

            computing = true;

            Trace.beginSection("imageAvailable");

            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            image.close();
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            LOGGER.e(e, "Exception!");
            Trace.endSection();
            return;
        }

        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

                        final long startTime = SystemClock.uptimeMillis();
                        stylizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        textureCopyBitmap = Bitmap.createBitmap(croppedBitmap);

                        requestRender();
                        computing = false;
                    }
                });

        Trace.endSection();
    }

    //Conversion between arrays of integers (provided by Android's getPixels() method) of
    // the form [0xRRGGBB, ...] to arrays of floats [-1.0, 1.0]
    // of the form [r, g, b, r, g, b, ...].
    private void stylizeImage(final Bitmap bitmap) {
        ++frameNum;
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        if (DEBUG_MODEL) {
            // Create a white square that steps through a black background 1 pixel per frame.
            final int centerX = (frameNum + bitmap.getWidth() / 2) % bitmap.getWidth();
            final int centerY = bitmap.getHeight() / 2;
            final int squareSize = 10;
            for (int i = 0; i < intValues.length; ++i) {
                final int x = i % bitmap.getWidth();
                final int y = i / bitmap.getHeight();
                final float val =
                        Math.abs(x - centerX) < squareSize && Math.abs(y - centerY) < squareSize ? 1.0f : 0.0f;
                floatValues[i * 3] = val;
                floatValues[i * 3 + 1] = val;
                floatValues[i * 3 + 2] = val;
            }
        } else {
            //final int test = 55;
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];

                floatValues[i * 3] = ((val >> 16) & 0xFF)  / (127.5f) - 1f ; //red

                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / (127.5f) - 1f; //green

                floatValues[i * 3 + 2] = (val & 0xFF) / (127.5f) - 1f; //blue

//                if(i == test){
//                    Log.d("ADebugTag", "\nFloatValue1: " + Float.toString(floatValues[i * 3]));
//                }
            }

//            // red
//            Log.d("ADebugTag", "\nValue1: " + Integer.toString((intValues[test] >> 16) & 0xFF));
//            Log.d("ADebugTag", "Altered Value1: " + Float.toString(floatValues[test * 3]));
//
//            //green
//            Log.d("ADebugTag", "\nValue2: " + Integer.toString((intValues[test] >> 8) & 0xFF));
//            Log.d("ADebugTag", "Altered Value2: " + Float.toString(floatValues[test * 3 + 1]));
//
//            //blue
//            Log.d("ADebugTag", "\nValue3: " + Integer.toString(intValues[test] & 0xFF));
//            Log.d("ADebugTag", "Altered Value3: " + Float.toString(floatValues[test * 3 + 2]));
        }

        //Pass the camera bitmap to TensorFlow then retrieve the graph output
        // Copy the input data into TensorFlow.
        inferenceInterface.feed(INPUT_NODE, floatValues,
                1, bitmap.getWidth(), bitmap.getHeight(), 3);

        // Execute the output node's dependency sub-graph.
        inferenceInterface.run(new String[] {OUTPUT_NODE}, isDebug());

        // Copy the data from TensorFlow back into our array.
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) ((floatValues[i * 3] + 1f) * 127.5f)) << 16) //red
                            | (((int) ((floatValues[i * 3 + 1] + 1f) * 127.5f)) << 8) //green
                            | ((int) ((floatValues[i * 3 + 2] + 1f) * 127.5f)); //blue
        }
        Log.d("ADebugTag", "\noutValue3: " + Integer.toString(((int) ((floatValues[55 * 3] + 1f) * 127.5f)) << 16));

        bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    }

    //Provides a debug overlay when you press the volume up or down buttons on the device, including
    // output from TensorFlow, performance metrics and the original, unprocessed, image.
    private void renderDebug(final Canvas canvas) {
        //get display height
        final DisplayMetrics metrics = new DisplayMetrics();
        Display display = getWindowManager().getDefaultDisplay();
        display.getRealMetrics(metrics);
        final float realHeight;
        realHeight = metrics.heightPixels;

        //get navigation bar height
        int navigationBarHeight = 0;
        int resourceId = getResources().getIdentifier("navigation_bar_height", "dimen", "android");
        if (resourceId > 0) {
            navigationBarHeight = getResources().getDimensionPixelSize(resourceId);
        }

        final Bitmap texture = textureCopyBitmap;
        if (texture != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor =
                    DEBUG_MODEL
                            ? 4.0f
                            : Math.min(
                            (float) canvas.getWidth() / texture.getWidth(),
                            (float) canvas.getHeight() / texture.getHeight());
            matrix.postScale(scaleFactor, scaleFactor);
            canvas.drawBitmap(texture, matrix, new Paint());

            final Bitmap copy1 = cropCopyBitmap;
            final Matrix matrix2 = new Matrix();
//            final float scaleFactor2 = 1;
            matrix2.postScale(scaleFactor, scaleFactor);
            matrix2.postTranslate(
//                    canvas.getWidth() - copy1.getWidth() * scaleFactor2,
                    1,
                    canvas.getHeight() - (realHeight/2 - navigationBarHeight)); //larger the num the higher up
            canvas.drawBitmap(copy1, matrix2, new Paint());
            Log.d("ADebugTag", "Altered Value3: " + Float.toString(texture.getHeight() * 3.3f));
        }

        if (!isDebug()) {
            return;
        }

        final Bitmap copy = cropCopyBitmap;

        if (copy == null) {
            return;
        }

        canvas.drawColor(0x55000000);

        final Vector<String> lines = new Vector<>();

        //Add TensorFlow status text to the overlay
        final String[] statLines = inferenceInterface.getStatString().split("\n");
        Collections.addAll(lines, statLines);
        lines.add("");


        lines.add("Frame: " + previewWidth + "x" + previewHeight);
        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
        lines.add("Rotation: " + sensorOrientation);
        lines.add("Inference time: " + lastProcessingTimeMs + "ms");
        lines.add("Desired size: " + desiredSize);
        lines.add("Initialized size: " + initializedSize);

        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
}