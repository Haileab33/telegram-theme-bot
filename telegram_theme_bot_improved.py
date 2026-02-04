"""
Telegram Theme Generator Bot

Creates custom Telegram themes for iOS, Android, and Desktop from photos.
Features:
- Extracts dominant colors from images using k-means clustering
- Generates matching UI colors with proper contrast
- Supports dark/light theme detection
- Multiple theme style options (vibrant, muted, pastel)

Commands:
/start - Welcome message and quick start guide
/help - Detailed help and tips
/description - Bot description and features
/generate - How to generate themes

To run: python telegram_theme_bot.py
Requires: pip install python-telegram-bot Pillow numpy
"""

import io
import json
import zipfile
import colorsys
import logging
from typing import List, Tuple, Dict, Optional
from PIL import Image
import numpy as np
from collections import Counter

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Bot Token - Replace with your actual token
import os
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Pillow resampling compatibility
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


# =============================================================================
# Color Utility Functions
# =============================================================================


def extract_dominant_colors(image: Image.Image, num_colors: int = 8) -> List[List[int]]:
    """
    Extract dominant colors using k-means clustering.
    
    Args:
        image: PIL Image to analyze
        num_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB color values sorted by frequency
    """
    if image.mode != "RGB":
        img = image.convert("RGB")
    else:
        img = image.copy()

    # Resize for faster processing while maintaining color accuracy
    img.thumbnail((400, 400), resample=LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    n_pixels = pixels.shape[0]
    if n_pixels == 0:
        return [[120, 120, 120]] * num_colors

    # Sample pixels for k-means
    sample_size = min(15000, n_pixels)
    if n_pixels > sample_size:
        idx = np.random.choice(n_pixels, sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    k = max(1, min(num_colors, len(sample)))

    # K-means clustering
    centers = sample[np.random.choice(len(sample), k, replace=False)].copy()

    for _ in range(15):  # Max iterations
        dists = np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)

        new_centers = []
        for i in range(k):
            members = sample[labels == i]
            if len(members) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(members.mean(axis=0))
        new_centers = np.array(new_centers)

        if np.allclose(new_centers, centers, atol=1.0):
            break
        centers = new_centers

    # Order by frequency
    final_dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    final_labels = final_dists.argmin(axis=1)
    counts = Counter(final_labels)
    ordered = [i for i, _ in counts.most_common()]

    dominant = [list(map(int, np.clip(centers[i], 0, 255))) for i in ordered]

    # Pad with grays if needed
    while len(dominant) < num_colors:
        dominant.append([120, 120, 120])

    return dominant[:num_colors]


def get_luminance(color: List[int]) -> float:
    """Calculate relative luminance (0-1 scale)."""
    r, g, b = [x / 255.0 for x in color[:3]]
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_saturation(color: List[int]) -> float:
    """Get saturation value of a color (0-1 scale)."""
    r, g, b = [x / 255.0 for x in color[:3]]
    _, _, s = colorsys.rgb_to_hls(r, g, b)
    return s


def adjust_lightness(color: List[int], amount: float) -> List[int]:
    """Adjust lightness of a color by amount (-1 to 1)."""
    r, g, b = [x / 255.0 for x in color[:3]]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + amount))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return [int(np.clip(c * 255, 0, 255)) for c in (r2, g2, b2)]


def adjust_saturation(color: List[int], amount: float) -> List[int]:
    """Adjust saturation of a color by amount (-1 to 1)."""
    r, g, b = [x / 255.0 for x in color[:3]]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s + amount))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return [int(np.clip(c * 255, 0, 255)) for c in (r2, g2, b2)]


def blend_colors(color1: List[int], color2: List[int], weight: float) -> List[int]:
    """Blend two colors. weight=0 gives color1, weight=1 gives color2."""
    return [
        int(np.clip(color1[i] * (1 - weight) + color2[i] * weight, 0, 255))
        for i in range(3)
    ]


def add_alpha(color: List[int], alpha: int) -> List[int]:
    """Add alpha channel to RGB color."""
    return [int(np.clip(x, 0, 255)) for x in color[:3]] + [int(np.clip(alpha, 0, 255))]


def contrast_ratio(color1: List[int], color2: List[int]) -> float:
    """Calculate WCAG contrast ratio between two colors."""
    l1 = get_luminance(color1)
    l2 = get_luminance(color2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def ensure_contrast(
    foreground: List[int], background: List[int], min_ratio: float = 4.5
) -> List[int]:
    """Adjust foreground color to ensure minimum contrast ratio with background."""
    current_ratio = contrast_ratio(foreground, background)
    
    if current_ratio >= min_ratio:
        return foreground
    
    bg_luminance = get_luminance(background)
    
    # Try lightening or darkening based on background
    if bg_luminance < 0.5:
        # Dark background - try lighter foreground
        for delta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            adjusted = adjust_lightness(foreground, delta)
            if contrast_ratio(adjusted, background) >= min_ratio:
                return adjusted
        return [255, 255, 255]  # Fallback to white
    else:
        # Light background - try darker foreground
        for delta in [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]:
            adjusted = adjust_lightness(foreground, delta)
            if contrast_ratio(adjusted, background) >= min_ratio:
                return adjusted
        return [0, 0, 0]  # Fallback to black


def rgb_to_hex(color: List[int]) -> str:
    """Convert RGB to hex string."""
    r, g, b = [int(np.clip(x, 0, 255)) for x in color[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def rgba_to_hex(color: List[int]) -> str:
    """Convert RGBA to hex string (RRGGBBAA or RRGGBB if fully opaque)."""
    r, g, b = [int(np.clip(x, 0, 255)) for x in color[:3]]
    if len(color) >= 4:
        a = int(np.clip(color[3], 0, 255))
        if a == 255:
            return f"#{r:02x}{g:02x}{b:02x}"
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    return f"#{r:02x}{g:02x}{b:02x}"


def to_argb_int(color: List[int], alpha: int = 255) -> int:
    """Convert color to signed 32-bit ARGB integer for Android themes."""
    if len(color) >= 4:
        alpha = color[3]
    r, g, b = [int(np.clip(x, 0, 255)) for x in color[:3]]
    alpha = int(np.clip(alpha, 0, 255))
    value = (alpha << 24) | (r << 16) | (g << 8) | b
    # Convert to signed 32-bit
    if value >= 0x80000000:
        value -= 0x100000000
    return value


# =============================================================================
# Theme Color Generation
# =============================================================================


def select_accent_color(colors: List[List[int]], is_dark: bool) -> List[int]:
    """Select the best accent color from dominant colors."""
    # Prefer saturated colors for accent
    scored = []
    for color in colors:
        sat = get_saturation(color)
        lum = get_luminance(color)
        # Prefer medium luminance and high saturation
        score = sat * 2 + (1 - abs(lum - 0.5))
        scored.append((score, color))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    
    # Get the most saturated color
    accent = scored[0][1] if scored else colors[0]
    
    # Ensure it's visible
    if is_dark and get_luminance(accent) < 0.3:
        accent = adjust_lightness(accent, 0.2)
    elif not is_dark and get_luminance(accent) > 0.7:
        accent = adjust_lightness(accent, -0.2)
    
    return accent


def generate_theme_colors(
    dominant_colors: List[List[int]], 
    is_dark: bool,
    style: str = "auto"
) -> Dict:
    """
    Generate complete theme color palette.
    
    Args:
        dominant_colors: List of dominant colors from image
        is_dark: Whether to generate dark theme
        style: Theme style - "auto", "vibrant", "muted", "pastel"
    """
    if not dominant_colors:
        dominant_colors = [[120, 120, 120]]
    
    primary = dominant_colors[0]
    secondary = dominant_colors[1] if len(dominant_colors) > 1 else primary
    tertiary = dominant_colors[2] if len(dominant_colors) > 2 else secondary
    quaternary = dominant_colors[3] if len(dominant_colors) > 3 else tertiary
    
    # Select accent color
    accent = select_accent_color(dominant_colors[:5], is_dark)
    
    # Apply style modifications
    if style == "vibrant":
        accent = adjust_saturation(accent, 0.3)
        primary = adjust_saturation(primary, 0.2)
    elif style == "muted":
        accent = adjust_saturation(accent, -0.2)
        primary = adjust_saturation(primary, -0.3)
    elif style == "pastel":
        accent = adjust_saturation(adjust_lightness(accent, 0.2), -0.3)
        primary = adjust_saturation(adjust_lightness(primary, 0.3), -0.4)
    
    if is_dark:
        # Dark theme palette
        text_primary = [255, 255, 255]
        text_secondary = [200, 200, 200]
        text_hint = [140, 140, 140]
        
        # Background colors
        background = [18, 18, 18]
        surface = [28, 28, 30]
        
        # Bubble colors - use primary color with adjustments
        incoming_base = blend_colors(primary, background, 0.7)
        incoming_bubble = add_alpha(adjust_lightness(incoming_base, -0.1), 200)
        
        outgoing_base = blend_colors(accent, background, 0.4)
        outgoing_bubble = add_alpha(adjust_lightness(outgoing_base, -0.05), 220)
        
        # Navigation
        nav_bar = add_alpha([22, 22, 24], 245)
        tab_bar = add_alpha([18, 18, 20], 250)
        
        # Input and separators
        input_bg = [38, 38, 40]
        separator = [55, 55, 58]
        
        # Name color - bright and contrasting
                # Name color - bright and contrasting
                # Name color - VERY bright and contrasting for Android visibility
        name_color = ensure_contrast(
            adjust_saturation(adjust_lightness(accent, 0.6), 0.4),  # 60% lighter, 40% more saturated
            incoming_bubble[:3],
            7.0  # Higher contrast ratio for better visibility
        )
        
        # Android-specific name colors - white for dark themes
        # Android-specific name colors - use accent color with good contrast for visibility
        # Instead of pure white, use a brighter version of accent for better visibility
        android_name_color = ensure_contrast(
            adjust_lightness(accent, 0.4),  # Make accent 40% lighter
            incoming_bubble[:3],
            4.5
        )
        
        # Link color
        link_color = adjust_lightness(accent, 0.15) if get_luminance(accent) < 0.5 else accent
        
    else:
        # Light theme palette
        text_primary = [0, 0, 0]
        text_secondary = [100, 100, 100]
        text_hint = [160, 160, 160]
        
        # Background colors
        background = [255, 255, 255]
        surface = [248, 248, 250]
        
        # Bubble colors
        incoming_bubble = add_alpha([255, 255, 255], 235)
        
        outgoing_base = adjust_lightness(accent, 0.25)
        outgoing_bubble = add_alpha(outgoing_base, 230)
        
        # Navigation
        nav_bar = add_alpha([250, 250, 252], 248)
        tab_bar = add_alpha([248, 248, 250], 252)
        
        # Input and separators
        input_bg = [242, 242, 247]
        separator = [220, 220, 225]
        
        # Name color
                # Name color - more vibrant for Android visibility
        name_color = ensure_contrast(
            adjust_lightness(accent, -0.3),  # 30% darker for better contrast
            incoming_bubble[:3],
            7.0  # Higher contrast ratio
        )
        
        # Link color
        link_color = adjust_lightness(accent, -0.1) if get_luminance(accent) > 0.6 else accent
    
    # Icon colors
    icon_active = accent
    icon_inactive = text_hint
    
    # Additional colors for extended palette
    badge_bg = accent
    badge_text = [255, 255, 255] if get_luminance(accent) < 0.6 else [0, 0, 0]
    
    return {
        # Primary colors
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
        "quaternary": quaternary,
        "accent": accent,
        
        # Text colors
        "text_primary": text_primary,
        "text_secondary": text_secondary,
        "text_hint": text_hint,
        
        # Background colors
        "background": background,
        "surface": surface,
        
        # Bubble colors
        "bubble_incoming": incoming_bubble,
        "bubble_outgoing": outgoing_bubble,
        
        # Navigation
        "nav_bar": nav_bar,
        "tab_bar": tab_bar,
        
        # Input
        "input_bg": input_bg,
        "separator": separator,
        
        # Special colors
        "name_color": name_color,
        "link_color": link_color,
        "icon_active": icon_active,
        "icon_inactive": icon_inactive,
        "badge_bg": badge_bg,
        "badge_text": badge_text,
    }


# =============================================================================
# Theme File Generators
# =============================================================================


def create_ios_theme(
    image: Image.Image, 
    theme_name: str = "CustomTheme",
    style: str = "auto"
) -> bytes:
    """Create iOS Telegram theme file (.tgios-theme)."""
    
    dominant_colors = extract_dominant_colors(image, 8)
    avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
    is_dark = avg_luminance < 0.5
    
    colors = generate_theme_colors(dominant_colors, is_dark, style)
    
    # Prepare wallpaper
    img = image.copy()
    img.thumbnail((1284, 2778), resample=LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    wallpaper_buffer = io.BytesIO()
    img.save(wallpaper_buffer, format="JPEG", quality=95)
    wallpaper_bytes = wallpaper_buffer.getvalue()
    
    # iOS theme JSON structure
    theme_settings = {
        "name": theme_name,
        "basedOn": "night" if is_dark else "day",
        "dark": is_dark,
        "colors": {
            # Navigation
            "rootNavigationBar": rgba_to_hex(colors["nav_bar"]),
            "navigationBar": rgba_to_hex(colors["nav_bar"]),
            "tabBar": rgba_to_hex(colors["tab_bar"]),
            
            # Accent
            "accent": rgb_to_hex(colors["accent"]),
            
            # Text
            "primaryText": rgb_to_hex(colors["text_primary"]),
            "secondaryText": rgb_to_hex(colors["text_secondary"]),
            
            # Chat
            "chatIncomingBubble": rgba_to_hex(colors["bubble_incoming"]),
            "chatOutgoingBubble": rgba_to_hex(colors["bubble_outgoing"]),
            "chatIncomingText": rgb_to_hex(colors["text_primary"]),
            "chatOutgoingText": rgb_to_hex([255, 255, 255] if is_dark else colors["text_primary"]),
            "chatIncomingAccent": rgb_to_hex(colors["name_color"]),
            "chatOutgoingAccent": rgb_to_hex(colors["name_color"]),
            
            # Input
            "inputBackground": rgb_to_hex(colors["input_bg"]),
            "inputText": rgb_to_hex(colors["text_primary"]),
            "inputPlaceholder": rgb_to_hex(colors["text_hint"]),
            
            # Other
            "separator": rgb_to_hex(colors["separator"]),
            "link": rgb_to_hex(colors["link_color"]),
            "badge": rgb_to_hex(colors["badge_bg"]),
            "badgeText": rgb_to_hex(colors["badge_text"]),
        },
        "wallpaper": "wallpaper.jpg"
    }
    
    # Create ZIP file
    theme_buffer = io.BytesIO()
    with zipfile.ZipFile(theme_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wallpaper.jpg", wallpaper_bytes)
        zf.writestr("theme.json", json.dumps(theme_settings, indent=2))
    
    return theme_buffer.getvalue()


def create_android_theme(
    image: Image.Image, 
    theme_name: str = "CustomTheme",
    style: str = "auto"
) -> bytes:
    """Create Android Telegram theme file (.attheme)."""
    
    dominant_colors = extract_dominant_colors(image, 8)
    avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
    is_dark = avg_luminance < 0.5
    
    colors = generate_theme_colors(dominant_colors, is_dark, style)
    
    # Android theme properties - comprehensive list
    theme_lines = [
        # Action bar
        f"actionBarDefault={to_argb_int(colors['nav_bar'])}",
        f"actionBarDefaultIcon={to_argb_int(colors['text_primary'])}",
        f"actionBarDefaultTitle={to_argb_int(colors['text_primary'])}",
        f"actionBarDefaultSubtitle={to_argb_int(colors['text_secondary'])}",
        f"actionBarDefaultSelector={to_argb_int(colors['accent'], 40)}",
        
        # Chat bubbles
        f"chat_inBubble={to_argb_int(colors['bubble_incoming'])}",
        f"chat_inBubbleSelected={to_argb_int(adjust_lightness(colors['bubble_incoming'][:3], -0.1))}",
        f"chat_inBubbleShadow={to_argb_int([0, 0, 0], 20)}",
        f"chat_outBubble={to_argb_int(colors['bubble_outgoing'])}",
        f"chat_outBubbleSelected={to_argb_int(adjust_lightness(colors['bubble_outgoing'][:3], -0.1))}",
        f"chat_outBubbleShadow={to_argb_int([0, 0, 0], 20)}",
        
        # Chat text
        f"chat_messageTextIn={to_argb_int(colors['text_primary'])}",
        f"chat_messageTextOut={to_argb_int([255, 255, 255] if is_dark else colors['text_primary'])}",
        f"chat_messageLinkIn={to_argb_int(colors['link_color'])}",
        f"chat_messageLinkOut={to_argb_int(colors['link_color'])}",
        f"chat_inVoiceSeekbar={to_argb_int(colors['name_color'])}",  # Often used for names
        f"chat_outVoiceSeekbar={to_argb_int(colors['name_color'])}",
        f"chat_selectedBackground={to_argb_int(colors['name_color'], 40)}",
        
        # Chat names - UPDATED SECTION
        f"chat_inNameText={to_argb_int(colors['name_color'])}",
        f"chat_outNameText={to_argb_int(colors['name_color'])}",
        f"chat_inAdminText={to_argb_int(colors['name_color'])}",
        f"chat_outAdminText={to_argb_int(colors['name_color'])}",
        f"chat_inReplyNameText={to_argb_int(colors['name_color'])}",
        f"chat_outReplyNameText={to_argb_int(colors['name_color'])}",
        f"chat_inNameIcon={to_argb_int(colors['name_color'])}",
        f"chat_outNameIcon={to_argb_int(colors['name_color'])}",
        f"chat_inForwardedNameText={to_argb_int(colors['name_color'])}",
        f"chat_outForwardedNameText={to_argb_int(colors['name_color'])}",
        
        # Message panel
        f"chat_messagePanelBackground={to_argb_int(colors['input_bg'])}",
        f"chat_messagePanelText={to_argb_int(colors['text_primary'])}",
        f"chat_messagePanelHint={to_argb_int(colors['text_hint'])}",
        f"chat_messagePanelSend={to_argb_int(colors['accent'])}",
        f"chat_messagePanelIcons={to_argb_int(colors['icon_inactive'])}",
        
        # Window backgrounds
        f"windowBackgroundWhite={to_argb_int(colors['background'])}",
        f"windowBackgroundGray={to_argb_int(colors['surface'])}",
        
        # Dialogs/Chats list
        f"chats_actionBackground={to_argb_int(colors['accent'])}",
        f"chats_actionIcon={to_argb_int(colors['badge_text'])}",
        f"chats_menuBackground={to_argb_int(colors['nav_bar'])}",
        f"chats_menuItemText={to_argb_int(colors['text_primary'])}",
        f"chats_nameMessage={to_argb_int(colors['text_secondary'])}",
        f"chats_unreadCounter={to_argb_int(colors['badge_bg'])}",
        f"chats_unreadCounterText={to_argb_int(colors['badge_text'])}",
        f"chats_nameArchived={to_argb_int(colors['name_color'])}",
        
        # Dividers
        f"divider={to_argb_int(colors['separator'])}",
        
        # List selector
        f"listSelectorSDK21={to_argb_int(colors['accent'], 30)}",
        
        # Switch
        f"switchTrack={to_argb_int(colors['text_hint'])}",
        f"switchTrackChecked={to_argb_int(colors['accent'])}",
        
        # Avatar
        f"avatar_nameIn={to_argb_int(colors['name_color'])}",
        f"avatar_nameOut={to_argb_int(colors['name_color'])}",
        f"chats_name={to_argb_int(colors['name_color'])}",
        f"chats_nameIcon={to_argb_int(colors['name_color'])}"
        f"avatar_backgroundRed={to_argb_int([255, 90, 90])}",
        f"avatar_backgroundOrange={to_argb_int([255, 150, 60])}",
        f"avatar_backgroundViolet={to_argb_int([150, 100, 255])}",
        f"avatar_backgroundGreen={to_argb_int([80, 200, 120])}",
        f"avatar_backgroundCyan={to_argb_int([80, 200, 200])}",
        f"avatar_backgroundBlue={to_argb_int([80, 140, 255])}",
        f"avatar_backgroundPink={to_argb_int([255, 120, 180])}",
    ]
    
    theme_content = "\n".join(theme_lines)
    
    # Prepare wallpaper
    img = image.copy()
    img.thumbnail((1080, 1920), resample=LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    wallpaper_buffer = io.BytesIO()
    img.save(wallpaper_buffer, format="JPEG", quality=95)
    wallpaper_bytes = wallpaper_buffer.getvalue()
    
    # Create .attheme file
    theme_buffer = io.BytesIO()
    theme_buffer.write(theme_content.encode("utf-8"))
    theme_buffer.write(b"\nWPS\n")
    theme_buffer.write(wallpaper_bytes)
    
    return theme_buffer.getvalue()


def create_desktop_theme(
    image: Image.Image, 
    theme_name: str = "CustomTheme",
    style: str = "auto"
) -> bytes:
    """Create Desktop Telegram theme file (.tdesktop-theme)."""
    
    dominant_colors = extract_dominant_colors(image, 8)
    avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
    is_dark = avg_luminance < 0.5
    
    colors = generate_theme_colors(dominant_colors, is_dark, style)
    
    # Desktop theme palette format
    palette_lines = [
        f"windowBg: {rgb_to_hex(colors['background'])};",
        f"windowFg: {rgb_to_hex(colors['text_primary'])};",
        f"windowBgOver: {rgb_to_hex(colors['surface'])};",
        f"windowBgRipple: {rgb_to_hex(adjust_lightness(colors['surface'], -0.05))};",
        f"windowFgOver: {rgb_to_hex(colors['text_primary'])};",
        f"windowSubTextFg: {rgb_to_hex(colors['text_secondary'])};",
        f"windowSubTextFgOver: {rgb_to_hex(colors['text_secondary'])};",
        f"windowBoldFg: {rgb_to_hex(colors['text_primary'])};",
        f"windowBoldFgOver: {rgb_to_hex(colors['text_primary'])};",
        f"windowBgActive: {rgb_to_hex(colors['accent'])};",
        f"windowFgActive: {rgb_to_hex(colors['badge_text'])};",
        f"windowActiveTextFg: {rgb_to_hex(colors['link_color'])};",
        f"windowShadowFg: {rgb_to_hex([0, 0, 0])};",
        f"windowShadowFgFallback: {rgb_to_hex([0, 0, 0])};",
        "",
        f"shadowFg: #00000020;",
        f"slideFadeOutBg: #0000003c;",
        f"slideFadeOutShadowFg: #00000000;",
        "",
        f"msgInBg: {rgba_to_hex(colors['bubble_incoming'])};",
        f"msgInBgSelected: {rgba_to_hex(add_alpha(adjust_lightness(colors['bubble_incoming'][:3], -0.08), colors['bubble_incoming'][3] if len(colors['bubble_incoming']) > 3 else 255))};",
        f"msgOutBg: {rgba_to_hex(colors['bubble_outgoing'])};",
        f"msgOutBgSelected: {rgba_to_hex(add_alpha(adjust_lightness(colors['bubble_outgoing'][:3], -0.08), colors['bubble_outgoing'][3] if len(colors['bubble_outgoing']) > 3 else 255))};",
        f"msgInServiceFg: {rgb_to_hex(colors['name_color'])};",
        f"msgInServiceFgSelected: {rgb_to_hex(colors['name_color'])};",
        f"msgOutServiceFg: {rgb_to_hex(colors['name_color'])};",
        f"msgOutServiceFgSelected: {rgb_to_hex(colors['name_color'])};",
        "",
        f"historyTextInFg: {rgb_to_hex(colors['text_primary'])};",
        f"historyTextOutFg: {rgb_to_hex([255, 255, 255] if is_dark else colors['text_primary'])};",
        f"historyLinkInFg: {rgb_to_hex(colors['link_color'])};",
        f"historyLinkOutFg: {rgb_to_hex(colors['link_color'])};",
        "",
        f"historyComposeAreaBg: {rgb_to_hex(colors['input_bg'])};",
        f"historyComposeAreaFg: {rgb_to_hex(colors['text_primary'])};",
        f"historyComposeAreaFgService: {rgb_to_hex(colors['text_hint'])};",
        "",
        f"titleBg: {rgba_to_hex(colors['nav_bar'])};",
        f"titleFg: {rgb_to_hex(colors['text_primary'])};",
        f"titleFgActive: {rgb_to_hex(colors['text_primary'])};",
        "",
        f"dialogsBg: {rgb_to_hex(colors['background'])};",
        f"dialogsNameFg: {rgb_to_hex(colors['text_primary'])};",
        f"dialogsTextFg: {rgb_to_hex(colors['text_secondary'])};",
        f"dialogsUnreadBg: {rgb_to_hex(colors['badge_bg'])};",
        f"dialogsUnreadFg: {rgb_to_hex(colors['badge_text'])};",
    ]
    
    palette_content = "\n".join(palette_lines)
    
    # Prepare wallpaper
    img = image.copy()
    img.thumbnail((1920, 1200), resample=LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    wallpaper_buffer = io.BytesIO()
    img.save(wallpaper_buffer, format="JPEG", quality=95)
    wallpaper_bytes = wallpaper_buffer.getvalue()
    
    # Create ZIP file
    theme_buffer = io.BytesIO()
    with zipfile.ZipFile(theme_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("colors.tdesktop-palette", palette_content)
        zf.writestr("background.jpg", wallpaper_bytes)
    
    return theme_buffer.getvalue()


# =============================================================================
# Bot Handlers
# =============================================================================


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    welcome_message = """🎨 **Telegram Theme Generator**

Welcome! I create beautiful custom themes from your photos.

**Quick Start:**
📸 Just send me a photo and I'll create a matching theme!

**Features:**
• iOS, Android & Desktop themes
• Smart color extraction
• Auto dark/light detection
• Multiple style options

**Commands:**
/start - This message
/help - Detailed help
/description - About this bot
/generate - How to create themes

Ready? Send me a photo! 🖼️"""
    
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """📖 **Help Guide**

**Creating a Theme:**
1️⃣ Send any photo to the bot
2️⃣ Choose your platform (iOS/Android/Desktop/All)
3️⃣ Optionally select a style
4️⃣ Download and apply your theme!

**Platform Details:**
• 📱 **iOS** - `.tgios-theme` file
• 🤖 **Android** - `.attheme` file  
• 💻 **Desktop** - `.tdesktop-theme` file

**Style Options:**
• **Auto** - Balanced colors
• **Vibrant** - Bold, saturated colors
• **Muted** - Subtle, professional
• **Pastel** - Soft, light colors

**Tips for Best Results:**
✅ Use high-resolution images
✅ Photos with distinct colors work great
✅ Landscapes & art often produce beautiful themes
✅ Both dark & light images supported

**Installing Themes:**
1. Download the theme file
2. Open it with Telegram
3. Preview and apply!

Need more help? Just ask! 💬"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def description_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /description command."""
    description_text = """ℹ️ **About This Bot**

**Telegram Theme Generator** transforms your photos into stunning Telegram themes.

**How It Works:**
1. 🔍 Analyzes your image using advanced color extraction
2. 🎨 Identifies 8 dominant colors using k-means clustering
3. 💡 Detects if the image is dark or light
4. 🖌️ Generates a harmonious color palette
5. ✨ Creates theme files with optimal contrast and readability

**Technology:**
• K-means color clustering
• WCAG contrast ratio optimization
• Adaptive brightness adjustment
• Cross-platform compatibility

**Supported Platforms:**
• Telegram iOS (iPhone/iPad)
• Telegram Android
• Telegram Desktop (Windows/Mac/Linux)

**Features:**
• Wallpaper integration
• Automatic theme type detection
• Multiple style presets
• Readable text colors
• Harmonious UI colors

Created with ❤️ for Telegram users worldwide!"""
    
    await update.message.reply_text(description_text, parse_mode="Markdown")


async def generate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /generate command."""
    generate_text = """🔧 **How to Generate Themes**

**Step 1: Choose Your Photo**
Send me any image - the better the quality, the better the theme!

**Great photos for themes:**
• 🌅 Landscapes & nature
• 🎨 Art & illustrations  
• 🏙️ City views
• 🌸 Flowers & patterns
• 📷 Your favorite photos

**Step 2: Select Platform**
After sending your photo, choose:
• **iOS** - For iPhone/iPad
• **Android** - For Android devices
• **Desktop** - For Windows/Mac/Linux
• **All** - Get all three!

**Step 3: Pick a Style** (Optional)
• **Auto** - Let the bot decide
• **Vibrant** - Bright, bold colors
• **Muted** - Subtle, calm tones
• **Pastel** - Soft, gentle hues

**Step 4: Apply Your Theme**
1. Download the theme file
2. Tap to open in Telegram
3. Preview the theme
4. Apply and enjoy! 🎉

**Pro Tips:**
💡 Dark images → Dark themes
💡 Light images → Light themes
💡 Colorful images → Most interesting themes

Ready? Just send a photo! 📸"""
    
    await update.message.reply_text(generate_text, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos."""
    try:
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎨 Received your photo! Analyzing colors..."
        )
        
        # Download photo
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        
        # Store for later
        context.user_data["pending_image"] = bytes(photo_bytes)
        
        # Analyze image for preview
        image = Image.open(io.BytesIO(photo_bytes))
        dominant_colors = extract_dominant_colors(image, 5)
        avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
        theme_type = "🌙 Dark" if avg_luminance < 0.5 else "☀️ Light"
        
        # Create color preview
        color_preview = " ".join([rgb_to_hex(c) for c in dominant_colors[:5]])
        
        await processing_msg.delete()
        
        # Platform selection keyboard
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📱 iOS", callback_data="PLATFORM_ios"),
                InlineKeyboardButton("🤖 Android", callback_data="PLATFORM_android"),
            ],
            [
                InlineKeyboardButton("💻 Desktop", callback_data="PLATFORM_desktop"),
                InlineKeyboardButton("📦 All", callback_data="PLATFORM_all"),
            ],
        ])
        
        preview_message = f"""✨ **Image Analyzed!**

**Detected:** {theme_type} theme
**Dominant Colors:**
{color_preview}

**Select your platform:**"""
        
        await update.message.reply_text(
            preview_message, 
            parse_mode="Markdown",
            reply_markup=keyboard
        )
        
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await update.message.reply_text(
            "❌ Failed to process the image. Please try again with a different photo."
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document uploads (images sent as files)."""
    document = update.message.document
    
    if document.mime_type and document.mime_type.startswith("image/"):
        try:
            processing_msg = await update.message.reply_text(
                "🎨 Receiving your image..."
            )
            
            file = await context.bot.get_file(document.file_id)
            photo_bytes = await file.download_as_bytearray()
            context.user_data["pending_image"] = bytes(photo_bytes)
            
            # Analyze image
            image = Image.open(io.BytesIO(photo_bytes))
            dominant_colors = extract_dominant_colors(image, 5)
            avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
            theme_type = "🌙 Dark" if avg_luminance < 0.5 else "☀️ Light"
            color_preview = " ".join([rgb_to_hex(c) for c in dominant_colors[:5]])
            
            await processing_msg.delete()
            
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📱 iOS", callback_data="PLATFORM_ios"),
                    InlineKeyboardButton("🤖 Android", callback_data="PLATFORM_android"),
                ],
                [
                    InlineKeyboardButton("💻 Desktop", callback_data="PLATFORM_desktop"),
                    InlineKeyboardButton("📦 All", callback_data="PLATFORM_all"),
                ],
            ])
            
            preview_message = f"""✨ **Image Analyzed!**

**Detected:** {theme_type} theme
**Dominant Colors:**
{color_preview}

**Select your platform:**"""
            
            await update.message.reply_text(
                preview_message,
                parse_mode="Markdown", 
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"Error handling document: {e}")
            await update.message.reply_text(
                "❌ Failed to process the image. Please try again."
            )
    else:
        await update.message.reply_text(
            "📸 Please send me a photo to create a theme!"
        )


async def handle_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle platform selection callback."""
    query = update.callback_query
    await query.answer()
    
    data = query.data or ""
    if not data.startswith("PLATFORM_"):
        return
    
    platform = data.replace("PLATFORM_", "")
    context.user_data["selected_platform"] = platform
    
    # Style selection keyboard
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎯 Auto", callback_data="STYLE_auto"),
            InlineKeyboardButton("🌈 Vibrant", callback_data="STYLE_vibrant"),
        ],
        [
            InlineKeyboardButton("🎨 Muted", callback_data="STYLE_muted"),
            InlineKeyboardButton("🌸 Pastel", callback_data="STYLE_pastel"),
        ],
    ])
    
    await query.message.edit_text(
        "🎨 **Choose a style:**\n\n"
        "• **Auto** - Balanced, natural colors\n"
        "• **Vibrant** - Bold, saturated look\n"
        "• **Muted** - Subtle, professional\n"
        "• **Pastel** - Soft, gentle tones",
        parse_mode="Markdown",
        reply_markup=keyboard
    )


async def handle_style_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle style selection and generate themes."""
    query = update.callback_query
    await query.answer()
    
    data = query.data or ""
    if not data.startswith("STYLE_"):
        return
    
    style = data.replace("STYLE_", "")
    platform = context.user_data.get("selected_platform", "all")
    pending = context.user_data.pop("pending_image", None)
    
    if not pending:
        await query.message.edit_text(
            "❌ No image found. Please send a photo first."
        )
        return
    
    await query.message.edit_text("⏳ Generating your theme(s)... Please wait!")
    
    try:
        image = Image.open(io.BytesIO(pending))
        
        # Generate theme info for caption
        dominant_colors = extract_dominant_colors(image, 5)
        avg_luminance = sum(get_luminance(c) for c in dominant_colors) / len(dominant_colors)
        theme_type = "🌙 Dark" if avg_luminance < 0.5 else "☀️ Light"
        color_preview = " ".join([rgb_to_hex(c) for c in dominant_colors[:5]])
        
        style_names = {
            "auto": "Auto",
            "vibrant": "Vibrant",
            "muted": "Muted",
            "pastel": "Pastel"
        }
        
        theme_name = f"CustomTheme_{query.from_user.id}"
        
        caption = f"""✨ **Your Custom Theme**

**Type:** {theme_type}
**Style:** {style_names.get(style, "Auto")}

**Colors Used:**
{color_preview}

Enjoy your new theme! 🎉"""
        
        # Generate requested themes
        if platform in ("ios", "all"):
            ios_bytes = create_ios_theme(image, theme_name, style)
            ios_file = io.BytesIO(ios_bytes)
            ios_file.name = f"{theme_name}.tgios-theme"
            await query.message.reply_document(
                document=ios_file,
                caption=caption if platform == "ios" else "📱 iOS Theme",
                parse_mode="Markdown"
            )
        
        if platform in ("android", "all"):
            android_bytes = create_android_theme(image, theme_name, style)
            android_file = io.BytesIO(android_bytes)
            android_file.name = f"{theme_name}.attheme"
            await query.message.reply_document(
                document=android_file,
                caption=caption if platform == "android" else "🤖 Android Theme",
                parse_mode="Markdown"
            )
        
        if platform in ("desktop", "all"):
            desktop_bytes = create_desktop_theme(image, theme_name, style)
            desktop_file = io.BytesIO(desktop_bytes)
            desktop_file.name = f"{theme_name}.tdesktop-theme"
            await query.message.reply_document(
                document=desktop_file,
                caption=caption if platform == "desktop" else "💻 Desktop Theme",
                parse_mode="Markdown"
            )
        
        await query.message.delete()
        
    except Exception as e:
        logger.error(f"Error generating theme: {e}")
        await query.message.edit_text(
            "❌ Failed to generate theme. Please try again with a different image."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages."""
    await update.message.reply_text(
        "📸 Please send me a photo to create a custom Telegram theme!\n\n"
        "Use /help for more information."
    )


def main() -> None:
    """Start the bot."""
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("❌ BOT_TOKEN environment variable is not set!")
        logger.error("Please set it in GitHub Secrets or .env file")
        return
    
    application = Application.builder().token(token).build()
    # ... rest of your code
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("description", description_command))
    application.add_handler(CommandHandler("generate", generate_command))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Callback handlers
    application.add_handler(CallbackQueryHandler(handle_platform_selection, pattern="^PLATFORM_"))
    application.add_handler(CallbackQueryHandler(handle_style_selection, pattern="^STYLE_"))
    
    logger.info("Starting Telegram Theme Generator Bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
