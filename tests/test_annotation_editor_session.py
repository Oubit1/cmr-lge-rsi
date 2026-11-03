from __future__ import annotations

import unittest
from pathlib import Path
from lxml import html


PROJECT = Path(__file__).resolve().parents[1]


class AnnotationEditorSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (PROJECT / "static/annotation_editor.html").read_text(encoding="utf-8")

    def test_draft_and_conversation_are_browser_persistent(self) -> None:
        self.assertIn("cmr-lge-editor:conversation:v2", self.source)
        self.assertIn("cmr-lge-editor:draft:v2", self.source)
        self.assertIn("function saveDraft()", self.source)
        self.assertIn("initConversation();", self.source)

    def test_case_navigation_does_not_clear_prompt_or_chat(self) -> None:
        self.assertNotIn("resetChat();$('note').value=''", self.source)
        self.assertNotIn("function resetChat()", self.source)
        self.assertIn("announceConversationContext();", self.source)

    def test_action_description_is_collapsed_by_default(self) -> None:
        self.assertIn("$('ctx-box').open=false", self.source)
        self.assertNotIn("$('ctx-box').open=true", self.source)

    def test_ui_surfaces_the_assigned_specialist(self) -> None:
        self.assertIn("out.assigned_role", self.source)
        self.assertIn("框架只负责路由，不代替专科判断", self.source)

    def test_case_double_click_copies_local_sample_reference(self) -> None:
        self.assertIn("el.ondblclick", self.source)
        self.assertIn("copyCaseReference", self.source)
        self.assertIn("Myo 标签路径", self.source)
        self.assertIn("navigator.clipboard.writeText", self.source)
        self.assertIn("复核状态：", self.source)
        self.assertNotIn("池/索引：", self.source)

    def test_internal_pool_selector_is_not_operator_facing(self) -> None:
        self.assertIn('id="pool" hidden', self.source)
        self.assertIn('id="boundary-filter" hidden', self.source)
        self.assertNotIn('>复核状态<', self.source)
        self.assertNotIn('>列表筛选<', self.source)
        self.assertNotIn("<label>Issue 池</label>", self.source)

    def test_boundary_review_has_four_simple_folder_states(self) -> None:
        for label in ("待筛", "待处理", "垃圾桶", "checked"):
            self.assertIn(label, self.source)
        self.assertIn("/api/editor/boundary-status", self.source)
        self.assertIn("group_spatial=${groupBoundarySpatial?'1':'0'}", self.source)

    def test_manual_ui_defaults_to_auto_note_and_reversible_removal(self) -> None:
        self.assertIn('id="remove-reason"', self.source)
        self.assertIn('id="restore-item"', self.source)
        self.assertIn('/api/editor/exclusion', self.source)
        self.assertIn('id="manual-note"', self.source)
        self.assertIn('tool_trace:toolTrace', self.source)
        self.assertNotIn('id="apply-code"', self.source)
        self.assertIn('id="editor-settings"', self.source)
        self.assertIn('class="workspace-top"', self.source)
        self.assertIn('const saved=await stageEdit();', self.source)
        self.assertIn("const code='APPLY';", self.source)
        self.assertIn('resizeCanvas();fitView();rebuildComposite();updateStats();draw();', self.source)

    def test_prompt_actions_share_bottom_save_and_apply(self) -> None:
        self.assertIn('id="prompt-run" onclick="requestPromptRepair(\'render\')"', self.source)
        self.assertIn('id="prompt-refine" disabled onclick="requestPromptRepair(\'refine\')"', self.source)
        self.assertNotIn('id="accept-prompt"', self.source)
        self.assertNotIn('id="draft-state"', self.source)
        self.assertIn('class="save-actions"', self.source)
        self.assertIn('Worked for ${Math.round', self.source)
        self.assertIn("mode==='refine'?activePrompt", self.source)

    def test_chat_and_manual_tools_are_parallel_without_mode_toggle(self) -> None:
        self.assertIn('<h1>CMR/LGE 标签编辑器</h1>', self.source)
        self.assertNotIn('标签修正助手', self.source)
        self.assertIn('<div class="chat-log" id="chat-log"></div>', self.source)
        self.assertIn("chatHistory.filter(entry=>entry.role==='user'||entry.role==='agent').slice(-6)", self.source)
        self.assertIn('min-height:110px;max-height:30vh', self.source)
        self.assertIn('<div class="manual-drawer" id="manual-drawer">', self.source)
        self.assertNotIn('id="manual-mode"', self.source)
        self.assertNotIn('function toggleManual()', self.source)
        self.assertNotIn('manualEnabled', self.source)

    def test_manual_drawing_prevents_late_candidate_overwrite(self) -> None:
        self.assertIn('manualRevision++', self.source)
        self.assertIn('if(promptJob!==jobId||manualRevision!==revision)break;', self.source)
        self.assertIn("markDirty('prompt')", self.source)
        self.assertIn('let promptBusy=false;', self.source)

    def test_official_iconfont_collection_is_used_as_decorative_icons(self) -> None:
        self.assertIn('https://www.iconfont.cn/collections/detail?cid=19238', self.source)
        for name in ('change', 'edit', 'delete', 'save', 'confirm', 'editor-eraser'):
            self.assertIn(f'id="i-{name}"', self.source)
            self.assertIn(f'href="#i-{name}"', self.source)
        self.assertIn('class="ui-icon" aria-hidden="true"', self.source)

    def test_chat_hides_navigation_noise_and_scrolls_after_render(self) -> None:
        self.assertIn("entry.role==='user'||entry.role==='agent'", self.source)
        self.assertNotIn('已切换样本：', self.source)
        self.assertIn('requestAnimationFrame(()=>{log.scrollTop=log.scrollHeight})', self.source)
        self.assertIn('persistChat();renderChat()', self.source)

    def test_both_polygon_tools_can_close_without_duplicate_vertices(self) -> None:
        self.assertIn('function polygonShouldClose(p)', self.source)
        self.assertIn('if(polygonShouldClose(p)){applyPoly();return}', self.source)
        self.assertIn('if(e.detail>=2&&poly.length>=3){applyPoly();return}', self.source)
        self.assertIn('tx.closePath();tx.fillStyle', self.source)
        self.assertIn("tool==='poly-erase'?0:1", self.source)
        self.assertIn('Math.abs(twiceArea)<1', self.source)

    def test_eraser_shortcuts_do_not_paint_on_right_click(self) -> None:
        self.assertIn("function toggleEraser(){setTool(tool==='erase'?'add':'erase')}", self.source)
        self.assertIn('if(e.button===2){e.preventDefault();toggleEraser();return}', self.source)
        self.assertIn("if(k==='e'&&!e.ctrlKey&&!e.metaKey&&!e.altKey)", self.source)
        self.assertIn('if(!e.repeat)toggleEraser()', self.source)
        self.assertIn("canvas.addEventListener('contextmenu',e=>e.preventDefault())", self.source)

    def test_reference_navigation_retains_unsaved_current_frame(self) -> None:
        self.assertIn('function holdCurrentEdit()', self.source)
        self.assertIn('function restoreHeldEdit()', self.source)
        self.assertIn('if(!isRef())holdCurrentEdit()', self.source)
        self.assertIn('if(!ref)restoreHeldEdit()', self.source)
        self.assertIn('function hasUnsavedWork(){return dirtyFlag||!!(heldEdit&&heldEdit.dirtyFlag)}', self.source)
        self.assertIn("if(isRef())return msg('请先回到当前帧，确认保留的修改后再应用。'", self.source)

    def test_agent_analysis_cannot_crowd_out_manual_tools(self) -> None:
        self.assertIn('max-height:5.5em;overflow:auto', self.source)
        self.assertIn("msg('候选已保存；Agent 分析完成。详细记录可悬停查看。'", self.source)
        self.assertIn('确认后看板会核对源标签、备份并写入正式标签', self.source)
        self.assertIn('out.audit_warning', self.source)

    def test_layout_keeps_canvas_and_right_editor_in_three_column_grid(self) -> None:
        document = html.fromstring(self.source)
        layout = document.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " layout ")]')
        self.assertEqual(len(layout), 1)
        self.assertEqual([child.tag for child in layout[0]], ['aside', 'main', 'aside'])
        self.assertEqual(layout[0][1].get('id'), 'workspace')
        self.assertIn('right', layout[0][2].get('class', '').split())

    def test_header_groups_navigation_and_review_judgment(self) -> None:
        document = html.fromstring(self.source)
        header = document.xpath('//header[contains(concat(" ", normalize-space(@class), " "), " top ")]')[0]
        self.assertEqual([child.get('class') for child in header], ['top-brand', 'top-navigation', 'review-bar'])
        self.assertTrue(header.xpath('./nav//*[@id="back-current"]'))
        self.assertTrue(header.xpath('./div[@id="review-bar"]//*[@id="boundary-controls"]'))
        self.assertFalse(document.xpath('//aside//*[@id="boundary-controls"]'))
        self.assertIn('flex:1;min-height:0', self.source)
        self.assertNotIn('id="review-bar" style="display:none"', self.source)
        self.assertIn("button.disabled=!boundary", self.source)
        self.assertIn('grid-template-columns:var(--side-width) minmax(0,1fr) var(--right-width)', self.source)
        self.assertIn('.top-navigation{justify-content:center;justify-self:stretch;grid-column:2}', self.source)
        self.assertIn('.review-bar{width:100%;grid-column:3;', self.source)

    def test_case_and_canvas_present_chinese_issue_without_bottom_evidence(self) -> None:
        document = html.fromstring(self.source)
        self.assertTrue(document.xpath('//*[@id="workspace-context"]'))
        self.assertIn('class="case-issue muted">${issueHeading(x.issues)}：', self.source)
        self.assertIn("codes.length===1&&codes[0]==='cmr_myo_small_area'?'筛查提示':'问题'", self.source)
        self.assertIn("cmr_tiny_unsupported_component:'离散小标注缺少相邻帧支持'", self.source)
        self.assertIn('证据：${evidenceText(current.evidence)}', self.source)
        self.assertIn('`标注分成 ${count} 块`', self.source)
        self.assertIn("${t==='myo'?'Myo Mask':'MI Mask'} 面积", self.source)
        self.assertNotIn('Issue：${current.issues}', self.source)

    def test_typography_uses_shared_body_button_and_title_tokens(self) -> None:
        self.assertIn('--font-ui:"Times New Roman","Microsoft YaHei",serif', self.source)
        self.assertIn('font:var(--body-size)/1.55 var(--font-ui)', self.source)
        self.assertIn('font:400 var(--button-size)/1.45 var(--font-ui)', self.source)
        self.assertIn('h1{font-size:var(--title-size)', self.source)

    def test_navigation_shows_one_layer_and_frame_label_with_jump_on_click(self) -> None:
        document = html.fromstring(self.source)
        self.assertFalse(document.xpath('//nav//b[text()="层" or text()="帧"]'))
        for dimension in ('layer', 'frame'):
            self.assertEqual(len(document.xpath(f'//button[@id="{dimension}-badge"]')), 1)
            self.assertTrue(document.xpath(f'//input[@id="{dimension}-jump" and @hidden]'))
        self.assertIn('function showJump(dim)', self.source)
        self.assertIn('function hideJump(dim)', self.source)

    def test_review_verdict_and_applied_edit_have_distinct_checked_paths(self) -> None:
        self.assertIn('aria-label="无需修改，进入 checked"', self.source)
        self.assertIn('aria-label="无价值，移入垃圾桶"', self.source)
        self.assertNotIn('onclick="focusForEditing()"', self.source)
        self.assertIn("if(status==='checked'&&staged)", self.source)
        self.assertIn("out.boundary_review_status==='checked'", self.source)
        self.assertIn('人工修改已应用；该空间层已自动进入 checked', self.source)

    def test_compact_header_buttons_keep_short_visible_labels(self) -> None:
        document = html.fromstring(self.source)
        header = document.xpath('//header[@class="top"]')[0]
        labels = ["".join(button.itertext()).strip() for button in header.xpath('.//button')]
        for label in ('当前', '通过', '剔除', 'ⓘ 说明'):
            self.assertIn(label, labels)
        self.assertNotIn('修改', labels)
        self.assertIn('看板', "".join(header.xpath('./div[@class="top-brand"]/a')[0].itertext()))

    def test_dashboard_bucket_link_can_open_a_filtered_review_queue(self) -> None:
        self.assertIn("params.get('review_status')", self.source)
        self.assertIn("['pending','work','checked','trash']", self.source)
        self.assertIn("$('boundary-filter').value=initialReviewStatus", self.source)


if __name__ == "__main__":
    unittest.main()
