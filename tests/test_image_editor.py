"""Tests for image_editor — stable key generation and helper functions."""

from unittest.mock import MagicMock, patch


class TestDisplayMetadata:
    def test_load_display_retries_after_metadata_error(self):
        """Metadata load failures should not mark display state initialized."""
        from streamtex import image_editor

        fake_st = MagicMock()
        fake_st.session_state = {}

        with patch.object(image_editor, "st", fake_st), \
             patch("streamtex.ai.history.get_current_metadata", side_effect=RuntimeError("boom")):
            image_editor._load_display_from_metadata("hero", "stx_img_display_hero")

        assert "stx_img_display_hero_initialized" not in fake_st.session_state

    def test_load_display_marks_initialized_after_success(self):
        """Successful metadata loads should initialize session display settings once."""
        from streamtex import image_editor
        from streamtex.ai.metadata import ImageMetadata

        fake_st = MagicMock()
        fake_st.session_state = {}
        meta = ImageMetadata(
            name="hero",
            display_zoom=60,
            display_width="70%",
            display_height="auto",
            display_keep_ratio=False,
        )

        with patch.object(image_editor, "st", fake_st), \
             patch("streamtex.ai.history.get_current_metadata", return_value=meta):
            image_editor._load_display_from_metadata("hero", "stx_img_display_hero")

        assert fake_st.session_state["stx_img_display_hero_initialized"] is True
        assert fake_st.session_state["stx_img_display_hero_zoom"] == 60
        assert fake_st.session_state["stx_img_display_hero_width"] == "70%"
        assert fake_st.session_state["stx_img_display_hero_height"] == "auto"
        assert fake_st.session_state["stx_img_display_hero_keep_ratio"] is False


class TestStableKey:
    def test_deterministic(self):
        """Same inputs produce same key."""
        from streamtex.image_editor import _stable_key
        k1 = _stable_key("img", "a", "b")
        k2 = _stable_key("img", "a", "b")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        """Different inputs produce different keys."""
        from streamtex.image_editor import _stable_key
        k1 = _stable_key("img", "a", "b")
        k2 = _stable_key("img", "c", "d")
        assert k1 != k2

    def test_prefix_included(self):
        """Key starts with the prefix."""
        from streamtex.image_editor import _stable_key
        k = _stable_key("myprefix", "x")
        assert k.startswith("myprefix_")

    def test_hash_length(self):
        """Key hash part is 10 characters."""
        from streamtex.image_editor import _stable_key
        k = _stable_key("p", "x")
        # Format: p_<10 hex chars>
        assert len(k) == len("p_") + 10

    def test_handles_non_string_parts(self):
        """Non-string parts are converted without error."""
        from streamtex.image_editor import _stable_key
        k = _stable_key("img", 42, None, 3.14)
        assert k.startswith("img_")


class TestDownloadUrlImage:
    def test_successful_download(self):
        """Successful download returns a temp file path."""
        from streamtex.image_editor import _download_url_image

        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG\r\n\x1a\n"
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = _download_url_image("https://example.com/img.png")
        assert result is not None
        assert result.endswith(".png")

        import os
        if result and os.path.exists(result):
            os.unlink(result)

    def test_jpeg_content_type(self):
        """JPEG content type produces .jpeg extension."""
        from streamtex.image_editor import _download_url_image

        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xd8\xff"
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = _download_url_image("https://example.com/photo.jpg")
        assert result is not None
        assert result.endswith(".jpeg")

        import os
        if result and os.path.exists(result):
            os.unlink(result)

    def test_failed_download_returns_none(self):
        """Network error returns None."""
        from streamtex.image_editor import _download_url_image

        with patch("requests.get", side_effect=Exception("Network error")):
            result = _download_url_image("https://example.com/img.png")
        assert result is None


class TestRenderEditorPanel:
    def test_skips_when_st_none(self):
        """_render_editor_panel returns early if st is None."""
        from streamtex import image_editor

        original_st = image_editor.st
        try:
            image_editor.st = None
            # Should not raise
            image_editor._render_editor_panel(
                uri="test.png", name="test", prompt=None,
                provider=None, model=None, ai_size=None,
                quality="standard", style=MagicMock(),
                width="100%", height="auto", alt="",
                link="", hover=True, light_bg=False,
            )
        finally:
            image_editor.st = original_st

    def test_skips_during_export(self, mock_streamlit):
        """_render_editor_panel returns early during export."""
        from streamtex.image_editor import _render_editor_panel

        with patch("streamtex.image_editor._is_exporting", return_value=True, create=True), \
             patch("streamtex.export._is_exporting", return_value=True, create=True):
            # Should not create expander
            _render_editor_panel(
                uri="test.png", name="test", prompt=None,
                provider=None, model=None, ai_size=None,
                quality="standard", style=MagicMock(),
                width="100%", height="auto", alt="",
                link="", hover=True, light_bg=False,
            )
