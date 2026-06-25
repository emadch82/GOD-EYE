use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn resize_image_bytes(_py: Python, frame_data: &[u8], target_w: u32, target_h: u32) -> PyResult<Py<PyBytes>> {
    let img = image::load_from_memory(frame_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Image decode error: {}", e)))?;
    let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let bytes = rgb.into_raw();
    Ok(PyBytes::new(_py, &bytes).into())
}

#[pyfunction]
fn decode_to_rgb(_py: Python, frame_data: &[u8]) -> PyResult<(u32, u32, Py<PyBytes>)> {
    let img = image::load_from_memory(frame_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Image decode error: {}", e)))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let bytes = rgb.into_raw();
    Ok((w, h, PyBytes::new(_py, &bytes).into()))
}

#[pyfunction]
fn preprocess_for_yolo(_py: Python, frame_data: &[u8], input_size: u32) -> PyResult<Py<PyBytes>> {
    let img = image::load_from_memory(frame_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Image decode error: {}", e)))?;
    let resized = img.resize_exact(input_size, input_size, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let raw = rgb.into_raw();
    let mut buf = Vec::with_capacity(raw.len() * 4);
    for &b in &raw {
        buf.extend_from_slice(&(b as f32 / 255.0).to_le_bytes());
    }
    Ok(PyBytes::new(_py, &buf).into())
}

#[pyfunction]
fn nms(boxes: Vec<(i32, i32, i32, i32, f32)>, threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].4.partial_cmp(&boxes[a].4).unwrap());
    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];
    for &i in &indices {
        if suppressed[i] { continue; }
        keep.push(i);
        for &j in &indices {
            if j == i || suppressed[j] { continue; }
            let iou = compute_iou(&boxes[i], &boxes[j]);
            if iou > threshold { suppressed[j] = true; }
        }
    }
    keep
}

fn compute_iou(a: &(i32, i32, i32, i32, f32), b: &(i32, i32, i32, i32, f32)) -> f32 {
    let x1 = a.0.max(b.0) as f32;
    let y1 = a.1.max(b.1) as f32;
    let x2 = a.2.min(b.2) as f32;
    let y2 = a.3.min(b.3) as f32;
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.2 - a.0) as f32 * (a.3 - a.1) as f32;
    let area_b = (b.2 - b.0) as f32 * (b.3 - b.1) as f32;
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[pyfunction]
fn crop_face(_py: Python, frame_data: &[u8], x1: i32, y1: i32, x2: i32, y2: i32) -> PyResult<Py<PyBytes>> {
    let img = image::load_from_memory(frame_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Image decode error: {}", e)))?;
    let rgb = img.to_rgb8();
    let w = rgb.width() as i32;
    let h = rgb.height() as i32;
    let x1 = x1.max(0).min(w) as u32;
    let y1 = y1.max(0).min(h) as u32;
    let x2 = x2.max(0).min(w) as u32;
    let y2 = y2.max(0).min(h) as u32;
    if x2 <= x1 || y2 <= y1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid crop region"));
    }
    let crop = image::imageops::crop_imm(&rgb, x1, y1, x2 - x1, y2 - y1).to_image();
    let mut buf = std::io::Cursor::new(Vec::new());
    crop.write_to(&mut buf, image::ImageFormat::Jpeg)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JPEG encode error: {}", e)))?;
    Ok(PyBytes::new(_py, buf.into_inner().as_slice()).into())
}

#[pymodule]
fn ai_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_image_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(decode_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_for_yolo, m)?)?;
    m.add_function(wrap_pyfunction!(nms, m)?)?;
    m.add_function(wrap_pyfunction!(crop_face, m)?)?;
    Ok(())
}
