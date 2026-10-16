'use strict';

const config = window.BEJEPA_CONFIG || {};

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function webUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

if (config.authors?.length) {
  const authors = config.authors.map(author => {
    const url = webUrl(author.url);
    const node = element(url ? 'a' : 'span', author.name);
    if (url) node.href = url;
    const labels = [...(author.affiliations || [])];
    if (author.corresponding) labels.push('†');
    if (labels.length) node.append(element('sup', labels.join(',')));
    return node;
  });
  document.querySelector('#authors').replaceChildren(...authors);
}

if (config.affiliations) {
  const affiliations = config.affiliations.map(affiliation => {
    const node = element('span');
    if (affiliation.id) node.append(element('sup', affiliation.id));
    node.append(document.createTextNode(affiliation.name));
    return node;
  });
  document.querySelector('#affiliations').replaceChildren(...affiliations);
}

// Leave the citation empty until the arXiv release is ready.
if (config.citation) document.querySelector('#bibtex').textContent = config.citation;

// Activate the paper button only when the actual preprint URL is supplied.
const arxivUrl = webUrl(config.arxivUrl);
if (arxivUrl) {
  const link = document.querySelector('#arxiv-link');
  link.href = arxivUrl;
  link.target = '_blank';
  link.rel = 'noopener';
  link.removeAttribute('aria-disabled');
  link.removeAttribute('role');
  link.title = 'Read the paper on arXiv';
  document.querySelector('#arxiv-status').hidden = true;
}

const video = document.querySelector('#project-video');
document.addEventListener('visibilitychange', () => {
  if (document.hidden) video.pause();
});
