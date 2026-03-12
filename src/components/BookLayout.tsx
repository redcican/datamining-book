import type { TocEntry, PageData } from '@/lib/types'
import Sidebar from './Sidebar'
import InPageTOC from './InPageTOC'
import PageNav from './PageNav'

export default function BookLayout({ toc, page }: { toc: TocEntry[]; page: PageData }) {
    const isDraft = page.frontmatter.status === 'draft'

    return (
        <div className="min-h-screen flex">
            {/* ── Left sidebar ──────────────────────────────────────────────── */}
            <Sidebar toc={toc} />

            {/* ── Main scroll area ──────────────────────────────────────────── */}
            <main
                className="flex-1 overflow-y-auto"
                style={{ marginLeft: 'var(--sidebar-w)' }}
            >
                <div
                    className="mx-auto py-12 px-8"
                    style={{ maxWidth: 'calc(var(--content-max) + var(--toc-w) + 2rem)' }}
                >
                    <article className="prose">
                        {isDraft && (
                            <div className="draft-banner">
                                🚧&nbsp; <strong>草稿</strong> — 本节内容正在编写中，尚未完成。
                            </div>
                        )}

                        {/* Render the HTML produced by unified pipeline */}
                        <div dangerouslySetInnerHTML={{ __html: page.contentHtml }} />
                    </article>

                    <PageNav prev={page.prev} next={page.next} />
                </div>
            </main>

            {/* ── Right in-page TOC ──────────────────────────────────────────── */}
            <InPageTOC headings={page.headings} />
        </div>
    )
}
