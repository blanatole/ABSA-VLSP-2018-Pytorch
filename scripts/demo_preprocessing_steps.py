#!/usr/bin/env python3
"""
Demo script để xuất kết quả từng bước tiền xử lý văn bản tiếng Việt
Hiển thị chi tiết quá trình xử lý text từ raw input đến final output
"""

import os
import sys
import re
import emoji
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from data_processing import (
        VietnameseTextCleaner, 
        VietnameseToneNormalizer, 
        VietnameseTextPreprocessor
    )
except ImportError:
    print("Warning: Could not import from src/data_processing.py")
    print("Using simplified preprocessing classes...")
    
    class SimpleVietnameseTextCleaner:
        VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
        
        @staticmethod
        def remove_html(text: str) -> str:
            return re.sub(r'<[^>]*>', '', text)
        
        @staticmethod
        def remove_emoji(text: str) -> str:
            try:
                return emoji.replace_emoji(text, '')
            except:
                return re.sub(r'[😀-🙏]', '', text)
        
        @staticmethod
        def remove_url(text: str) -> str:
            return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)
        
        @staticmethod
        def remove_email(text: str) -> str:
            return re.sub(r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+', '', text)
        
        @staticmethod
        def remove_phone_number(text: str) -> str:
            return re.sub(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$', '', text)
        
        @staticmethod
        def remove_hashtags(text: str) -> str:
            return re.sub(r'#\w+', '', text)
        
        @staticmethod
        def remove_unnecessary_characters(text: str) -> str:
            text = re.sub(fr"[^\sa-zA-Z0-9{SimpleVietnameseTextCleaner.VN_CHARS}]", ' ', text)
            return re.sub(r'\s+', ' ', text).strip()
        
        @staticmethod
        def process_text(text: str) -> str:
            """Apply all cleaning steps"""
            text = SimpleVietnameseTextCleaner.remove_html(text)
            text = SimpleVietnameseTextCleaner.remove_emoji(text)
            text = SimpleVietnameseTextCleaner.remove_url(text)
            text = SimpleVietnameseTextCleaner.remove_email(text)
            text = SimpleVietnameseTextCleaner.remove_phone_number(text)
            text = SimpleVietnameseTextCleaner.remove_hashtags(text)
            text = SimpleVietnameseTextCleaner.remove_unnecessary_characters(text)
            return text

    VietnameseTextCleaner = SimpleVietnameseTextCleaner


def print_step_header(step_num: int, step_name: str, description: str = ""):
    """In header cho mỗi bước"""
    print(f"\n{'='*80}")
    print(f"BƯỚC {step_num}: {step_name}")
    if description:
        print(f"📝 {description}")
    print('='*80)


def print_step_result(original: str, processed: str, step_name: str):
    """In kết quả của từng bước xử lý"""
    print(f"\n🔸 {step_name}:")
    print(f"   Trước: '{original}'")
    print(f"   Sau:   '{processed}'")
    
    if original != processed:
        print(f"   ✅ Có thay đổi")
    else:
        print(f"   ➡️  Không thay đổi")


def normalize_teencodes_simple(text: str) -> str:
    """Normalize teencode đơn giản"""
    teencodes = {
        'ks': 'khách sạn',
        'nhahang': 'nhà hàng', 
        'nv': 'nhân viên',
        'dv': 'dịch vụ',
        'pt': 'phòng tắm',
        'dt': 'điện thoại',
        'fb': 'facebook',
        'ok': 'được',
        'ko': 'không',
        'k': 'không',
        'tks': 'cảm ơn',
        'thanks': 'cảm ơn',
        'gud': 'tốt',
        'good': 'tốt',
        'bad': 'tệ',
        'wa': 'quá',
        'vs': 'với',
        'j': 'gì',
        'r': 'rồi',
        'dc': 'được',
        'đc': 'được'
    }
    
    words = text.split()
    processed_words = []
    
    for word in words:
        if word.lower() in teencodes:
            processed_words.append(teencodes[word.lower()])
        else:
            processed_words.append(word)
    
    return ' '.join(processed_words)


def normalize_tone_simple(text: str) -> str:
    """Normalize tone đơn giản - fix một số lỗi thường gặp"""
    # Một số normalization cơ bản cho tiếng Việt
    tone_fixes = {
        'lựơng': 'lượng',
        'thỏai': 'thoải', 
        'hoà': 'hòa',
        'khoá': 'khóa',
        'toà': 'tòa',
        'gia đình': 'gia đình',  # giữ nguyên
    }
    
    for wrong, correct in tone_fixes.items():
        text = text.replace(wrong, correct)
    
    return text


def segment_words_simple(text: str) -> str:
    """Word segmentation đơn giản - chỉ handle một số trường hợp cơ bản"""
    # Một số compound words tiếng Việt cần segment
    compound_words = {
        'khách sạn': 'khách_sạn',
        'nhà hàng': 'nhà_hàng',
        'nhân viên': 'nhân_viên',
        'dịch vụ': 'dịch_vụ',
        'phòng tắm': 'phòng_tắm',
        'điện thoại': 'điện_thoại',
        'cám ơn': 'cám_ơn',
        'cảm ơn': 'cảm_ơn',
        'không gian': 'không_gian',
        'thời gian': 'thời_gian',
        'chất lượng': 'chất_lượng',
        'giá cả': 'giá_cả',
        'vị trí': 'vị_trí',
        'máy lạnh': 'máy_lạnh'
    }
    
    for phrase, segmented in compound_words.items():
        text = text.replace(phrase, segmented)
    
    return text


def demo_preprocessing_pipeline(sample_texts: List[str], output_file: str = None):
    """Demo complete preprocessing pipeline với output chi tiết"""
    
    print("🚀 DEMO PIPELINE TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT")
    print("📅 Thời gian:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\n📋 Pipeline gồm các bước:")
    print("   1. Chuyển về chữ thường (Lowercase)")
    print("   2. Loại bỏ HTML tags")
    print("   3. Loại bỏ emoji")
    print("   4. Loại bỏ URL") 
    print("   5. Loại bỏ email")
    print("   6. Loại bỏ số điện thoại")
    print("   7. Loại bỏ hashtags")
    print("   8. Loại bỏ ký tự không cần thiết")
    print("   9. Chuẩn hóa tone tiếng Việt")
    print("  10. Chuẩn hóa teencode")
    print("  11. Phân đoạn từ (Word Segmentation)")
    
    results = []
    
    for idx, original_text in enumerate(sample_texts):
        print(f"\n\n{'🎯'*3} SAMPLE {idx + 1} {'🎯'*3}")
        print(f"📝 Text gốc: '{original_text}'")
        
        # Bước 1: Lowercase
        current_text = original_text
        step1_text = current_text.lower()
        print_step_header(1, "CHUYỂN CHỮ THƯỜNG", "Đưa tất cả về lowercase để chuẩn hóa")
        print_step_result(current_text, step1_text, "Lowercase")
        current_text = step1_text
        
        # Bước 2: Remove HTML
        step2_text = VietnameseTextCleaner.remove_html(current_text)
        print_step_header(2, "LOẠI BỎ HTML TAGS", "Xóa các tag HTML như <b>, <div>, etc.")
        print_step_result(current_text, step2_text, "Remove HTML")
        current_text = step2_text
        
        # Bước 3: Remove Emoji
        step3_text = VietnameseTextCleaner.remove_emoji(current_text)
        print_step_header(3, "LOẠI BỎ EMOJI", "Xóa các emoji và emoticon")
        print_step_result(current_text, step3_text, "Remove Emoji")
        current_text = step3_text
        
        # Bước 4: Remove URL
        step4_text = VietnameseTextCleaner.remove_url(current_text)
        print_step_header(4, "LOẠI BỎ URL", "Xóa các liên kết web")
        print_step_result(current_text, step4_text, "Remove URL")
        current_text = step4_text
        
        # Bước 5: Remove Email
        step5_text = VietnameseTextCleaner.remove_email(current_text)
        print_step_header(5, "LOẠI BỎ EMAIL", "Xóa địa chỉ email")
        print_step_result(current_text, step5_text, "Remove Email")
        current_text = step5_text
        
        # Bước 6: Remove Phone
        step6_text = VietnameseTextCleaner.remove_phone_number(current_text)
        print_step_header(6, "LOẠI BỎ SỐ ĐIỆN THOẠI", "Xóa số điện thoại")
        print_step_result(current_text, step6_text, "Remove Phone")
        current_text = step6_text
        
        # Bước 7: Remove Hashtags
        step7_text = VietnameseTextCleaner.remove_hashtags(current_text)
        print_step_header(7, "LOẠI BỎ HASHTAGS", "Xóa các hashtag #tag")
        print_step_result(current_text, step7_text, "Remove Hashtags")
        current_text = step7_text
        
        # Bước 8: Remove Unnecessary Characters
        step8_text = VietnameseTextCleaner.remove_unnecessary_characters(current_text)
        print_step_header(8, "LOẠI BỎ KÝ TỰ KHÔNG CẦN THIẾT", "Chỉ giữ chữ, số và dấu tiếng Việt")
        print_step_result(current_text, step8_text, "Clean Characters")
        current_text = step8_text
        
        # Bước 9: Normalize Tone
        step9_text = normalize_tone_simple(current_text)
        print_step_header(9, "CHUẨN HÓA TONE TIẾNG VIỆT", "Sửa lỗi gõ tone như lựơng -> lượng")
        print_step_result(current_text, step9_text, "Normalize Tone")
        current_text = step9_text
        
        # Bước 10: Normalize Teencodes
        step10_text = normalize_teencodes_simple(current_text)
        print_step_header(10, "CHUẨN HÓA TEENCODE", "Chuyển teencode thành từ chuẩn: ks -> khách sạn")
        print_step_result(current_text, step10_text, "Normalize Teencode")
        current_text = step10_text
        
        # Bước 11: Word Segmentation
        step11_text = segment_words_simple(current_text)
        print_step_header(11, "PHÂN ĐOẠN TỪ", "Ghép từ ghép: khách sạn -> khách_sạn")
        print_step_result(current_text, step11_text, "Word Segmentation")
        final_text = step11_text
        
        # Tổng kết
        print(f"\n{'🎉'*10} KẾT QUẢ CUỐI CÙNG {'🎉'*10}")
        print(f"📥 Input:  '{original_text}'")
        print(f"📤 Output: '{final_text}'")
        
        # Thống kê thay đổi
        words_before = len(original_text.split())
        words_after = len(final_text.split())
        chars_before = len(original_text)
        chars_after = len(final_text)
        
        print(f"\n📊 Thống kê:")
        print(f"   • Số từ: {words_before} → {words_after} ({'+'*(words_after-words_before) if words_after >= words_before else str(words_after-words_before)})")
        print(f"   • Số ký tự: {chars_before} → {chars_after} ({'+'*(chars_after-chars_before) if chars_after >= chars_before else str(chars_after-chars_before)})")
        
        results.append({
            'sample_id': idx + 1,
            'original_text': original_text,
            'step1_lowercase': step1_text,
            'step2_remove_html': step2_text,
            'step3_remove_emoji': step3_text,
            'step4_remove_url': step4_text,
            'step5_remove_email': step5_text,
            'step6_remove_phone': step6_text,
            'step7_remove_hashtags': step7_text,
            'step8_clean_chars': step8_text,
            'step9_normalize_tone': step9_text,
            'step10_normalize_teencode': step10_text,
            'step11_word_segment': step11_text,
            'final_output': final_text,
            'words_before': words_before,
            'words_after': words_after,
            'chars_before': chars_before,
            'chars_after': chars_after
        })
    
    # Xuất kết quả ra file CSV nếu được yêu cầu
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Đã lưu kết quả chi tiết vào: {output_file}")
    
    return results


def main():
    """Main function"""
    print("🇻🇳 DEMO TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT - ABSA VLSP 2018")
    print("=" * 80)
    
    # Sample texts cho demo
    sample_texts = [
        # Text với HTML và emoji
        "Ks này rất <b>đẹp</b> và sạch sẽ 😍! Nv thân thiện ok.",
        
        # Text với URL và email
        "Nhahang tốt, xem thêm tại https://example.com hoặc liên hệ test@gmail.com",
        
        # Text với hashtags và phone
        "Món ăn ngon wa #food #delicious. Gọi 0901234567 để đặt bàn.",
        
        # Text với lỗi tone và teencode
        "Chất lựơng ko tốt, thỏai mái vs giá cả hợp lý. Tks!",
        
        # Text phức tạp với nhiều vấn đề
        "Cám ơn shop 😊! Sản phẩm gud, ship nhanh. Visit: www.shop.com #shopping 👍",
        
        # Text thực tế từ review khách sạn
        "Phòng sạch sẽ, nhân viên dv tốt. Vị trí khách sạn thuận tiện, gần trung tâm.",
        
        # Text với nhiều ký tự đặc biệt
        "!!! Giá cả ok, chất lượng @#$% không như mong đợi !!! FB: hotelpage"
    ]
    
    # Chạy demo
    results = demo_preprocessing_pipeline(
        sample_texts, 
        output_file='preprocessing_demo_results.csv'
    )
    
    print(f"\n\n{'🔥'*20} TỔNG KẾT {'🔥'*20}")
    print(f"✅ Đã xử lý {len(sample_texts)} sample texts")
    print(f"📊 Kết quả tổng thể:")
    
    total_chars_before = sum(r['chars_before'] for r in results)
    total_chars_after = sum(r['chars_after'] for r in results)
    total_words_before = sum(r['words_before'] for r in results)
    total_words_after = sum(r['words_after'] for r in results)
    
    print(f"   • Tổng ký tự: {total_chars_before} → {total_chars_after}")
    print(f"   • Tổng từ: {total_words_before} → {total_words_after}")
    print(f"   • Tỷ lệ nén ký tự: {total_chars_after/total_chars_before:.2%}")
    print(f"   • Tỷ lệ thay đổi từ: {total_words_after/total_words_before:.2%}")
    
    print(f"\n🎯 Các bước quan trọng nhất:")
    print(f"   1. Chuẩn hóa teencode (ks → khách sạn)")
    print(f"   2. Loại bỏ ký tự không cần thiết")
    print(f"   3. Phân đoạn từ (khách sạn → khách_sạn)")
    print(f"   4. Chuẩn hóa tone tiếng Việt")
    
    print(f"\n📝 Lưu ý: Script này sử dụng preprocessing đơn giản.")
    print(f"   Trong thực tế, project sử dụng:")
    print(f"   • VnCoreNLP cho word segmentation chính xác")
    print(f"   • Model bmd1905/vietnamese-correction-v2 cho error correction")
    print(f"   • Teencode dictionary đầy đủ từ behitek")


if __name__ == "__main__":
    main() 